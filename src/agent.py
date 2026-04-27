"""Agentic pipeline.

Per-turn flow:
  1. parse_intent(query, history)         — LLM call #1, structured JSON
  2. recommend_songs(prefs, songs, k=10)  — deterministic scorer (tool)
  3. explain_recommendations(...)         — LLM call #2, grounded on tool output
  4. AgentTurn assembled and returned

The deterministic scorer in src/recommender.py is the source of truth for
ranking. The LLM never reorders. If the LLM picks IDs outside the scored
candidate set during explanation, we drop its output and fall back to the
deterministic reason strings — the agent can never invent songs.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import ollama
from pydantic import ValidationError

from src.data import get_allowed_genres, get_allowed_moods
from src.recommender import recommend_songs
from src.schemas import AgentTurn, IntentSchema, RecommendationCard

MODEL = os.environ.get("SOUNDFIT_MODEL", "qwen2.5:3b")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_OPTS = {"temperature": 0, "seed": 42, "top_p": 1.0}


class OllamaUnavailableError(RuntimeError):
    """Raised when the Ollama service is unreachable. The CLI/UI catches
    this and degrades to pure rule-based mode with a banner."""


def _client() -> ollama.Client:
    return ollama.Client(host=OLLAMA_HOST)


def _ollama_chat(messages: List[Dict[str, str]], format_json: bool = True) -> str:
    try:
        kwargs: Dict[str, Any] = {
            "model": MODEL,
            "messages": messages,
            "options": OLLAMA_OPTS,
        }
        if format_json:
            kwargs["format"] = "json"
        resp = _client().chat(**kwargs)
        return resp["message"]["content"]
    except (ConnectionError, OSError, ollama.ResponseError) as exc:
        raise OllamaUnavailableError(
            f"Cannot reach Ollama at {OLLAMA_HOST}: {exc}. "
            f"Start it with `docker compose up ollama` "
            f"or set OLLAMA_HOST to a remote instance."
        ) from exc
    except Exception as exc:  # connection-related httpx errors
        msg = str(exc).lower()
        if any(t in msg for t in ("connect", "refused", "timeout", "host", "resolve")):
            raise OllamaUnavailableError(
                f"Cannot reach Ollama at {OLLAMA_HOST}: {exc}"
            ) from exc
        raise


def _intent_system_prompt(allowed_genres: List[str], allowed_moods: List[str]) -> str:
    return f"""You are a music preference extractor. Read the user's chat
message and any prior turns and fill out a JSON object describing what
music they want. Use slang context to infer numeric features.

ALLOWED GENRES (use ONLY these — if user mentions one not in this list,
set genre=null; do NOT guess a similar one): {sorted(allowed_genres)}

ALLOWED MOODS (use ONLY these): {sorted(allowed_moods)}

NUMERIC FIELDS (0.0-1.0; null if query doesn't imply):
  energy: 0.0=very calm, 1.0=very energetic
  valence: 0.0=sad, 1.0=happy
  danceability: 0.0=hard to dance, 1.0=very danceable
  acousticness: 0.0=produced/electric, 1.0=acoustic

ARTIST — extract ONLY from the current message, NEVER from the chat
history. If the most recent user message explicitly asks for music BY a
specific artist ("songs by Drake", "play some Beyoncé", "Bad Bunny
tracks"), put that artist's name in `artist`. Otherwise set
`artist=null` — even if a previous turn mentioned an artist. Each turn's
artist filter is fresh.

If the user says "music LIKE X" or "similar to X", leave `artist` null
and set numeric features matching that artist's style instead.

COMMON SLANG → FEATURE MAPPINGS:
  "aux songs", "aux cord", "rolling music"  → energy >=0.8, mood=intense, often hip-hop or pop
  "workout", "gym", "run", "pumped"          → energy >=0.85, mood=intense
  "study", "coding", "focus", "background"   → mood=focused, energy <=0.4, acousticness>=0.5
  "chill", "vibing", "lo-fi", "lofi"         → mood=chill, energy 0.3-0.5, genre=electronic
  "hype", "turn up", "going out", "party"    → energy>=0.85, valence>=0.6, danceability>=0.7
  "heartbreak", "sad", "rainy", "crying"     → valence<=0.3, mood=moody
  "romance", "love", "cuddle"                → valence>=0.6, energy 0.3-0.6
  "throwback", "old school", "classic"       → no numeric mapping; rely on genre
  "hard", "raw", "trap"                      → energy>=0.8, mood=intense, genre=hip-hop
  "smooth", "easy", "mellow"                 → energy<=0.5, valence>=0.45, mood=chill or relaxed

REFINEMENT — set `refinement_of_previous=true` ONLY when the user is
explicitly modifying the previous request:
  IS refinement: "more upbeat", "less intense", "show different ones",
                 "what about jazz instead", "skip these", "more like #2"
  NOT refinement (set false): "give me songs for X", "recommend Y",
                              "now play Z", "songs by [different artist]",
                              any new vibe/scenario/artist request even
                              if it follows another music request.
A new scenario ("car ride music", "workout playlist", "study session")
is a FRESH query, not a refinement, even if the previous turn was about
music. When in doubt, choose false.

EXAMPLES (study these — your output must look like the right side):
  "aux rap songs"           → {{"genre":"hip-hop","mood":"intense","energy":0.85,"valence":0.55,"danceability":0.75}}
  "songs by The Weeknd"     → {{"artist":"The Weeknd"}}
  "chill lofi for studying" → {{"genre":"electronic","mood":"focused","energy":0.3,"acousticness":0.7}}
  "more upbeat though"      → {{"refinement_of_previous":true,"energy":0.85}}
  "play some Drake"         → {{"artist":"Drake"}}
  "polka music"             → {{"genre":null}}
  "sad bedroom pop"         → {{"genre":"pop","mood":"moody","valence":0.25,"energy":0.4}}
  "hype workout playlist"   → {{"mood":"intense","energy":0.9,"valence":0.65,"danceability":0.7}}

Return ONLY a JSON object with EXACTLY these fields (omit none, use null
where unset):
{{
  "genre": string or null,
  "mood": string or null,
  "artist": string or null,
  "energy": number or null,
  "valence": number or null,
  "danceability": number or null,
  "acousticness": number or null,
  "k": integer (default 5),
  "refinement_of_previous": boolean,
  "exclude_song_ids": array of integers
}}"""


def _validate_intent(
    raw_json: str,
    allowed_genres: List[str],
    allowed_moods: List[str],
    current_message: str = "",
) -> IntentSchema:
    """Parse + coerce + validate the LLM's JSON. Falls back to an empty
    IntentSchema on any failure so the pipeline degrades gracefully.

    Hard rule: `artist` is only kept if it appears (case-insensitive
    substring) in the current_message. Stops the LLM from carrying
    artists forward from chat history."""
    try:
        data = json.loads(raw_json)
    except (ValueError, TypeError):
        return IntentSchema()

    if not isinstance(data, dict):
        return IntentSchema()

    for field in ("energy", "valence", "danceability", "acousticness"):
        v = data.get(field)
        if isinstance(v, str):
            try:
                data[field] = float(v)
            except ValueError:
                data[field] = None
        elif isinstance(v, (int, float)):
            data[field] = max(0.0, min(1.0, float(v)))

    allowed_genre_lc = {g.lower() for g in allowed_genres}
    allowed_mood_lc = {m.lower() for m in allowed_moods}
    if data.get("genre"):
        if str(data["genre"]).strip().lower() not in allowed_genre_lc:
            data["genre"] = None
    if data.get("mood"):
        if str(data["mood"]).strip().lower() not in allowed_mood_lc:
            data["mood"] = None

    if data.get("artist") and current_message:
        artist_lc = str(data["artist"]).strip().lower()
        message_lc = current_message.lower()
        if artist_lc and artist_lc not in message_lc:
            # The LLM pulled this artist from history, not from the user's
            # current message. Drop it — each turn's artist is fresh.
            data["artist"] = None

    if not isinstance(data.get("exclude_song_ids"), list):
        data["exclude_song_ids"] = []
    else:
        data["exclude_song_ids"] = [
            int(x) for x in data["exclude_song_ids"] if isinstance(x, (int, float, str)) and str(x).lstrip("-").isdigit()
        ]

    try:
        return IntentSchema(**data)
    except ValidationError:
        return IntentSchema()


def parse_intent(
    query: str,
    history: Optional[List[Dict[str, Any]]] = None,
    allowed_genres: Optional[List[str]] = None,
    allowed_moods: Optional[List[str]] = None,
) -> IntentSchema:
    """LLM call #1 — structured intent extraction."""
    history = history or []
    allowed_genres = allowed_genres or []
    allowed_moods = allowed_moods or get_allowed_moods()

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": _intent_system_prompt(allowed_genres, allowed_moods)}
    ]
    for turn in history[-4:]:
        role = turn.get("role", "user")
        content = turn.get("content", "")
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": str(content)})
    messages.append({"role": "user", "content": query})

    raw = _ollama_chat(messages, format_json=True)
    return _validate_intent(raw, allowed_genres, allowed_moods, current_message=query)


def merge_intents(prev: IntentSchema, new: IntentSchema) -> IntentSchema:
    """Carry forward feature fields from `prev` for any field `new` left null,
    but only when the user signaled a refinement. `artist` is NOT carried —
    if a user wants the same artist on a follow-up they'll name them again,
    and stale artists from misclassified refinements just propagate fallback
    notes."""
    if not new.refinement_of_previous:
        return new
    merged = new.model_dump()
    prev_dump = prev.model_dump()
    for key in ("genre", "mood", "energy", "valence", "danceability", "acousticness"):
        if merged.get(key) is None and prev_dump.get(key) is not None:
            merged[key] = prev_dump[key]
    return IntentSchema(**merged)


def _filter_by_artist(songs: List[Dict[str, Any]], artist: str) -> List[Dict[str, Any]]:
    """Substring match (case-insensitive). 'weeknd' matches 'The Weeknd';
    'drake' matches 'Drake' but not 'Drake Bell'... well, it would match
    Drake Bell too — fine for our purposes. We accept false positives over
    false negatives so the user always gets something."""
    needle = artist.strip().lower()
    if not needle:
        return songs
    return [s for s in songs if needle in str(s.get("artist", "")).lower()]


def _explanation_system_prompt(artist_fallback: Optional[str] = None) -> str:
    artist_block = ""
    if artist_fallback:
        artist_block = f"""

CRITICAL — ARTIST FALLBACK MODE: The user asked for music by '{artist_fallback}'
but that artist is NOT in our catalog. The candidates below are similar-style
picks, NOT '{artist_fallback}' tracks.
- DO NOT call any candidate a '{artist_fallback}' song or track.
- DO NOT say 'this is {artist_fallback}'-style' unless the candidate's actual artist is given.
- DO frame each as a similar-style alternative based on its real features.
"""
    return f"""You write short, grounded explanations of why each candidate
song fits the user's request. Each explanation must cite at least one
concrete feature value (energy, valence, mood, genre, tempo, or
acousticness). One sentence, max 25 words.{artist_block}

GROUNDING RULES (these are HARD constraints):
- ONLY cite the candidate's actual artist, title, genre, mood, and feature values as given to you below.
- DO NOT invent any artist name, song title, or attribute.
- DO NOT cite an artist name from the user's query if that artist is not on the candidate row.
- If the candidate's artist field says "Justin Bieber", do not write that it's a Drake song.

Return ONLY a JSON object of this exact shape:
{{"explanations": {{"<song_id_int>": "<explanation>", ...}}}}

Use ONLY the integer song IDs you were given. Do not invent IDs."""


def _fallback_explanations(candidates: List[Dict[str, Any]]) -> Dict[int, str]:
    out: Dict[int, str] = {}
    for c in candidates:
        det = c.get("deterministic_reasons") or ""
        if det:
            out[int(c["id"])] = det
        else:
            out[int(c["id"])] = (
                f"{c.get('title')} by {c.get('artist')} — close fit on the catalog features."
            )
    return out


def explain_recommendations(
    query: str,
    candidates: List[Dict[str, Any]],
    artist_fallback: Optional[str] = None,
) -> Dict[int, str]:
    """LLM call #2 — short grounded explanation per candidate ID.

    Hard guarantee: if the LLM returns IDs outside the candidate set, or
    invalid JSON, or fails the schema, we return deterministic reason
    strings instead. The LLM cannot smuggle in songs that weren't scored.
    """
    expected_ids = {int(c["id"]) for c in candidates}

    candidate_lines = "\n".join(
        f"  id={c['id']} title={c['title']!r} artist={c['artist']!r} "
        f"genre={c['genre']} mood={c['mood']} "
        f"energy={float(c['energy']):.2f} valence={float(c['valence']):.2f} "
        f"danceability={float(c['danceability']):.2f} "
        f"acousticness={float(c['acousticness']):.2f} "
        f"tempo={float(c['tempo_bpm']):.0f}bpm"
        for c in candidates
    )
    user_prompt = (
        f"User request: {query}\n\n"
        f"Candidates (write one explanation per ID below):\n{candidate_lines}"
    )

    raw = _ollama_chat(
        [
            {"role": "system", "content": _explanation_system_prompt(artist_fallback)},
            {"role": "user", "content": user_prompt},
        ],
        format_json=True,
    )

    try:
        data = json.loads(raw)
        raw_map = data.get("explanations", {})
        if not isinstance(raw_map, dict):
            return _fallback_explanations(candidates)
        explanations: Dict[int, str] = {}
        for k, v in raw_map.items():
            try:
                key = int(k)
            except (TypeError, ValueError):
                continue
            if key in expected_ids and isinstance(v, str) and v.strip():
                explanations[key] = v.strip()
        if not set(explanations.keys()) >= expected_ids:
            fallback = _fallback_explanations(candidates)
            for k in expected_ids:
                explanations.setdefault(k, fallback[k])
        return explanations
    except (ValueError, TypeError):
        return _fallback_explanations(candidates)


def run_agent(
    query: str,
    songs: List[Dict[str, Any]],
    history: Optional[List[Dict[str, Any]]] = None,
    allowed_genres: Optional[List[str]] = None,
    allowed_moods: Optional[List[str]] = None,
    k: int = 5,
) -> AgentTurn:
    """Full per-turn agent pipeline. Returns an AgentTurn with cards.

    Always returns a usable result. Falls back to the rule-based scorer
    with deterministic reason strings when Ollama is unreachable."""
    history = history or []
    allowed_genres = allowed_genres or get_allowed_genres(songs)
    allowed_moods = allowed_moods or get_allowed_moods()

    notes: List[str] = []
    used_baseline = False

    try:
        intent = parse_intent(query, history, allowed_genres, allowed_moods)
    except OllamaUnavailableError as exc:
        notes.append(f"Ollama unavailable for intent parsing: {exc}")
        used_baseline = True
        intent = IntentSchema()

    if intent.refinement_of_previous and history:
        prev_intent = next(
            (
                turn["parsed_intent"]
                for turn in reversed(history)
                if isinstance(turn, dict) and turn.get("parsed_intent") is not None
            ),
            None,
        )
        if prev_intent is not None:
            if isinstance(prev_intent, dict):
                prev_intent = IntentSchema(**prev_intent)
            intent = merge_intents(prev_intent, intent)

    candidate_pool = [s for s in songs if int(s["id"]) not in set(intent.exclude_song_ids)]
    artist_fallback: Optional[str] = None
    if intent.artist:
        artist_pool = _filter_by_artist(candidate_pool, intent.artist)
        if artist_pool:
            candidate_pool = artist_pool
            notes.append(f"Filtered to {len(artist_pool)} tracks matching artist '{intent.artist}'.")
        else:
            artist_fallback = intent.artist
            notes.append(
                f"No tracks by '{intent.artist}' in catalog — showing similar-style picks instead."
            )

    prefs = intent.to_user_prefs()
    requested_k = intent.k or k
    pool_size = max(requested_k * 3, 10)
    ranked = recommend_songs(prefs, candidate_pool, k=pool_size)
    top_k = ranked[:requested_k]

    if not top_k:
        return AgentTurn(
            query=query,
            parsed_intent=intent,
            cards=[],
            used_baseline_fallback=used_baseline,
            notes=notes + ["No songs matched."],
        )

    candidate_dicts: List[Dict[str, Any]] = [
        {**song, "deterministic_reasons": reasons} for song, _, reasons in top_k
    ]

    if used_baseline:
        explanations = _fallback_explanations(candidate_dicts)
    else:
        try:
            explanations = explain_recommendations(
                query, candidate_dicts, artist_fallback=artist_fallback
            )
        except OllamaUnavailableError as exc:
            notes.append(f"Ollama dropped during explanation: {exc}")
            explanations = _fallback_explanations(candidate_dicts)
            used_baseline = True

    cards: List[RecommendationCard] = []
    for song, score, det_reasons in top_k:
        sid = int(song["id"])
        cards.append(
            RecommendationCard(
                song_id=sid,
                title=str(song["title"]),
                artist=str(song["artist"]),
                genre=str(song["genre"]),
                mood=str(song["mood"]),
                energy=float(song["energy"]),
                valence=float(song["valence"]),
                danceability=float(song["danceability"]),
                acousticness=float(song["acousticness"]),
                tempo_bpm=float(song["tempo_bpm"]),
                score=float(score),
                deterministic_reasons=det_reasons,
                explanation=explanations.get(sid, det_reasons or ""),
            )
        )

    return AgentTurn(
        query=query,
        parsed_intent=intent,
        cards=cards,
        used_baseline_fallback=used_baseline,
        notes=notes,
    )


def warmup() -> None:
    """Best-effort one-token call to mask cold-start latency. Silent on
    failure — startup should never block on this."""
    try:
        _client().generate(
            model=MODEL, prompt="hi", options={"num_predict": 1, "temperature": 0}
        )
    except Exception:
        pass
