"""Unit tests for src/agent.py — the parts that don't require a live Ollama.

Live-Ollama tests live in tests/test_agent_eval.py and are gated behind
the --ollama pytest flag (added in checkpoint 3)."""

import json

import pytest

from src.agent import (
    OllamaUnavailableError,
    _fallback_explanations,
    _validate_intent,
    explain_recommendations,
    merge_intents,
    run_agent,
)
from src.data import get_allowed_genres, get_allowed_moods, _load_sample_csv
from src.schemas import IntentSchema


# --- _validate_intent ---------------------------------------------------------


def test_validate_intent_parses_clean_json():
    raw = json.dumps({"genre": "pop", "mood": "happy", "energy": 0.8, "k": 5})
    intent = _validate_intent(raw, ["pop", "rock"], ["happy", "chill"])
    assert intent.genre == "pop"
    assert intent.mood == "happy"
    assert intent.energy == 0.8
    assert intent.k == 5


def test_validate_intent_nulls_oov_genre():
    """Critical safety property: OOV genres must NOT pass through."""
    raw = json.dumps({"genre": "polka", "mood": "happy"})
    intent = _validate_intent(raw, ["pop", "rock"], ["happy", "chill"])
    assert intent.genre is None
    assert intent.mood == "happy"


def test_validate_intent_nulls_oov_mood():
    raw = json.dumps({"genre": "pop", "mood": "ecstatic"})
    intent = _validate_intent(raw, ["pop"], ["happy", "chill"])
    assert intent.mood is None
    assert intent.genre == "pop"


def test_validate_intent_coerces_string_numerics():
    raw = json.dumps({"energy": "0.7", "valence": "0.5"})
    intent = _validate_intent(raw, [], [])
    assert intent.energy == 0.7
    assert intent.valence == 0.5


def test_validate_intent_clamps_out_of_range_numerics():
    raw = json.dumps({"energy": 1.5, "valence": -0.2})
    intent = _validate_intent(raw, [], [])
    assert intent.energy == 1.0
    assert intent.valence == 0.0


def test_validate_intent_returns_empty_on_garbage_json():
    intent = _validate_intent("not json at all", [], [])
    assert intent == IntentSchema()


def test_validate_intent_returns_empty_on_non_dict_json():
    intent = _validate_intent("[1,2,3]", [], [])
    assert intent == IntentSchema()


def test_validate_intent_drops_artist_not_in_current_message():
    """If the LLM hallucinates an artist (e.g. pulling 'Kanye West' from
    chat history while the user is asking about car-ride music), the
    validator must drop it. Hard guarantee, not vibes."""
    raw = json.dumps({"genre": "hip-hop", "artist": "Kanye West"})
    intent = _validate_intent(
        raw, ["hip-hop"], [], current_message="give me car ride aux songs"
    )
    assert intent.artist is None
    assert intent.genre == "hip-hop"  # other fields preserved


def test_validate_intent_keeps_artist_when_present_in_message():
    raw = json.dumps({"artist": "Bad Bunny"})
    intent = _validate_intent(raw, [], [], current_message="play some Bad Bunny tracks")
    assert intent.artist == "Bad Bunny"


def test_validate_intent_keeps_artist_with_partial_name_match():
    """User says 'Drake', LLM correctly extracts 'Drake' — keep it."""
    raw = json.dumps({"artist": "Drake"})
    intent = _validate_intent(raw, [], [], current_message="play drake")
    assert intent.artist == "Drake"


def test_validate_intent_strips_non_int_exclude_ids():
    raw = json.dumps({"exclude_song_ids": [1, "2", "abc", 3.5]})
    intent = _validate_intent(raw, [], [])
    assert 1 in intent.exclude_song_ids
    assert 2 in intent.exclude_song_ids
    assert "abc" not in [str(x) for x in intent.exclude_song_ids if isinstance(x, str)]


# --- merge_intents ------------------------------------------------------------


def test_merge_intents_fills_nulls_from_previous_when_refining():
    prev = IntentSchema(genre="pop", mood="happy", energy=0.8)
    new = IntentSchema(refinement_of_previous=True, energy=0.4)
    merged = merge_intents(prev, new)
    assert merged.genre == "pop"
    assert merged.mood == "happy"
    assert merged.energy == 0.4  # the refinement wins where it specifies


def test_merge_intents_does_not_carry_when_not_refining():
    prev = IntentSchema(genre="pop", mood="happy", energy=0.8)
    new = IntentSchema(genre="jazz")
    merged = merge_intents(prev, new)
    assert merged.genre == "jazz"
    assert merged.mood is None
    assert merged.energy is None


# --- _fallback_explanations ---------------------------------------------------


def test_fallback_uses_deterministic_reasons_when_present():
    candidates = [
        {"id": 1, "title": "X", "artist": "Y", "deterministic_reasons": "Genre match: pop"}
    ]
    res = _fallback_explanations(candidates)
    assert res[1] == "Genre match: pop"


def test_fallback_synthesizes_when_no_reasons():
    candidates = [{"id": 7, "title": "Foo", "artist": "Bar", "deterministic_reasons": ""}]
    res = _fallback_explanations(candidates)
    assert "Foo" in res[7] and "Bar" in res[7]


# --- explain_recommendations integrity guarantees ----------------------------


def test_explain_falls_back_when_llm_returns_invalid_ids(monkeypatch):
    """If the LLM returns IDs outside the candidate set, we must NOT pass
    them through. Hard guarantee — the agent never invents songs."""
    candidates = [{"id": 1, "title": "A", "artist": "X", "genre": "pop", "mood": "happy",
                   "energy": 0.8, "valence": 0.8, "danceability": 0.7,
                   "acousticness": 0.1, "tempo_bpm": 120,
                   "deterministic_reasons": "Genre match"}]
    monkeypatch.setattr(
        "src.agent._ollama_chat",
        lambda *a, **k: json.dumps({"explanations": {"999": "this id wasn't in the list"}}),
    )
    res = explain_recommendations("test query", candidates)
    # Only valid IDs survive; 999 is dropped, falls back to deterministic for ID 1
    assert 1 in res
    assert 999 not in res
    assert res[1] == "Genre match"


def test_explain_falls_back_on_invalid_json(monkeypatch):
    candidates = [{"id": 1, "title": "A", "artist": "X", "genre": "pop", "mood": "happy",
                   "energy": 0.5, "valence": 0.5, "danceability": 0.5,
                   "acousticness": 0.5, "tempo_bpm": 100,
                   "deterministic_reasons": "Mood match: happy"}]
    monkeypatch.setattr("src.agent._ollama_chat", lambda *a, **k: "not json at all")
    res = explain_recommendations("test query", candidates)
    assert res[1] == "Mood match: happy"


# --- run_agent end-to-end (with Ollama mocked out) ---------------------------


def test_run_agent_falls_back_when_ollama_down(monkeypatch):
    """If Ollama is unreachable, run_agent returns rule-based cards
    and sets used_baseline_fallback=True. No exception escapes."""
    from src import agent as agent_mod

    def boom(*args, **kwargs):
        raise OllamaUnavailableError("connection refused")

    monkeypatch.setattr(agent_mod, "parse_intent", boom)

    songs = _load_sample_csv().to_dict("records")
    turn = run_agent("workout music", songs)
    assert turn.used_baseline_fallback is True
    assert len(turn.cards) > 0
    assert all(isinstance(c.song_id, int) for c in turn.cards)


def test_run_agent_uses_intent_to_filter(monkeypatch):
    """When intent is parsed cleanly, the deterministic scorer ranks pop
    songs first for a pop intent."""
    from src import agent as agent_mod

    monkeypatch.setattr(
        agent_mod, "parse_intent",
        lambda q, h, ag, am: IntentSchema(genre="pop", mood="happy", energy=0.8),
    )
    monkeypatch.setattr(
        agent_mod, "explain_recommendations",
        lambda q, c, **kw: {int(x["id"]): "stub explanation" for x in c},
    )

    songs = _load_sample_csv().to_dict("records")
    turn = run_agent("happy pop music", songs)
    assert turn.used_baseline_fallback is False
    assert len(turn.cards) > 0
    assert turn.cards[0].genre == "pop"


def test_run_agent_respects_exclude_song_ids(monkeypatch):
    from src import agent as agent_mod

    songs = _load_sample_csv().to_dict("records")
    excluded_ids = [s["id"] for s in songs if s["genre"] == "pop"][:5]

    monkeypatch.setattr(
        agent_mod, "parse_intent",
        lambda q, h, ag, am: IntentSchema(
            genre="pop", mood="happy", energy=0.8,
            exclude_song_ids=excluded_ids,
        ),
    )
    monkeypatch.setattr(
        agent_mod, "explain_recommendations",
        lambda q, c, **kw: {int(x["id"]): "stub" for x in c},
    )

    turn = run_agent("happy pop music", songs)
    returned_ids = {c.song_id for c in turn.cards}
    assert returned_ids.isdisjoint(set(excluded_ids))


def test_run_agent_filters_by_artist_when_specified(monkeypatch):
    """When intent.artist is set and songs by that artist exist, the
    candidate pool should be restricted to that artist."""
    from src import agent as agent_mod

    monkeypatch.setattr(
        agent_mod, "parse_intent",
        lambda q, h, ag, am: IntentSchema(artist="The Weeknd"),
    )
    monkeypatch.setattr(
        agent_mod, "explain_recommendations",
        lambda q, c, **kw: {int(x["id"]): "stub" for x in c},
    )

    songs = _load_sample_csv().to_dict("records")
    turn = run_agent("songs by the weeknd", songs)
    assert len(turn.cards) > 0
    for card in turn.cards:
        assert "weeknd" in card.artist.lower()


def test_run_agent_falls_back_to_full_pool_when_artist_unknown(monkeypatch):
    """If no songs match the artist, the agent should still return cards
    rather than empty — better to show something close than nothing."""
    from src import agent as agent_mod

    monkeypatch.setattr(
        agent_mod, "parse_intent",
        lambda q, h, ag, am: IntentSchema(artist="Some Made Up Artist", genre="pop"),
    )
    monkeypatch.setattr(
        agent_mod, "explain_recommendations",
        lambda q, c, **kw: {int(x["id"]): "stub" for x in c},
    )

    songs = _load_sample_csv().to_dict("records")
    turn = run_agent("songs by some made up artist", songs)
    assert len(turn.cards) > 0
    assert any("not in catalog" in n.lower() or "no tracks" in n.lower() for n in turn.notes)


def test_run_agent_artist_case_insensitive(monkeypatch):
    from src import agent as agent_mod

    monkeypatch.setattr(
        agent_mod, "parse_intent",
        lambda q, h, ag, am: IntentSchema(artist="bad bunny"),
    )
    monkeypatch.setattr(
        agent_mod, "explain_recommendations",
        lambda q, c, **kw: {int(x["id"]): "stub" for x in c},
    )

    songs = _load_sample_csv().to_dict("records")
    turn = run_agent("bad bunny tracks", songs)
    assert len(turn.cards) > 0
    for card in turn.cards:
        assert "bad bunny" in card.artist.lower()


def test_run_agent_passes_artist_fallback_to_explainer(monkeypatch):
    """When the requested artist isn't in the catalog, run_agent must pass
    that artist's name to explain_recommendations as artist_fallback so the
    LLM is told NOT to claim candidates are by that artist."""
    from src import agent as agent_mod

    captured = {}

    def capture_explain(query, candidates, artist_fallback=None):
        captured["artist_fallback"] = artist_fallback
        return {int(c["id"]): "stub" for c in candidates}

    monkeypatch.setattr(
        agent_mod, "parse_intent",
        lambda q, h, ag, am: IntentSchema(artist="Some Made Up Artist", genre="pop"),
    )
    monkeypatch.setattr(agent_mod, "explain_recommendations", capture_explain)

    songs = _load_sample_csv().to_dict("records")
    run_agent("songs by some made up artist", songs)
    assert captured["artist_fallback"] == "Some Made Up Artist"


def test_run_agent_does_not_pass_fallback_when_artist_found(monkeypatch):
    """When the artist IS in the catalog, no fallback signal should be set."""
    from src import agent as agent_mod

    captured = {}

    def capture_explain(query, candidates, artist_fallback=None):
        captured["artist_fallback"] = artist_fallback
        return {int(c["id"]): "stub" for c in candidates}

    monkeypatch.setattr(
        agent_mod, "parse_intent",
        lambda q, h, ag, am: IntentSchema(artist="The Weeknd"),
    )
    monkeypatch.setattr(agent_mod, "explain_recommendations", capture_explain)

    songs = _load_sample_csv().to_dict("records")
    run_agent("songs by the weeknd", songs)
    assert captured["artist_fallback"] is None


def test_merge_intents_does_not_carry_artist_on_refinement():
    """Artist is intentionally NOT carried over even on refinements.
    A stale artist from a misclassified refinement re-triggers fallback
    notes on a fresh query — better to drop it. If the user wants the
    same artist, they'll name them again."""
    prev = IntentSchema(artist="The Weeknd", genre="r&b", energy=0.6)
    new = IntentSchema(refinement_of_previous=True, energy=0.85)
    merged = merge_intents(prev, new)
    assert merged.artist is None  # NOT carried
    assert merged.genre == "r&b"  # carried (genre is sticky)
    assert merged.energy == 0.85  # the refinement wins


def test_run_agent_never_returns_song_outside_catalog(monkeypatch):
    """Hardest guarantee: even when the LLM tries to hallucinate IDs,
    the cards we surface always come from the input catalog."""
    from src import agent as agent_mod

    monkeypatch.setattr(
        agent_mod, "parse_intent",
        lambda q, h, ag, am: IntentSchema(genre="pop"),
    )
    # LLM tries to inject a fake ID
    monkeypatch.setattr(
        "src.agent._ollama_chat",
        lambda *a, **k: json.dumps({"explanations": {"99999": "hallucinated"}}),
    )

    songs = _load_sample_csv().to_dict("records")
    catalog_ids = {s["id"] for s in songs}

    turn = run_agent("pop music", songs)
    for card in turn.cards:
        assert card.song_id in catalog_ids
