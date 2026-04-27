"""Streamlit chatbox UI for SoundFit.

Run with:
    streamlit run src/streamlit_app.py

Layout:
- Main column: chat with `st.chat_input` + `st.chat_message`. Each agent
  reply renders its top-K as compact cards with feature bars and the
  LLM-written explanation.
- Sidebar: controls (reset / baseline toggle / Spotify enrichment status)
  + tabs for `Trace` (last turn's parsed intent + candidate scores) and
  `Eval` (reads eval/results/summary.json + latest.csv to render the
  metrics dashboard).

State lives in st.session_state. The catalog and warmup are cached so
the app doesn't pay for them on every interaction.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

# Streamlit invokes `streamlit run src/streamlit_app.py` with `src/` on
# sys.path (not the project root), which breaks `from src.X` imports.
# Insert the repo root so the existing module layout works untouched.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd
import streamlit as st

from src.agent import OllamaUnavailableError, run_agent, warmup
from src.data import (
    get_allowed_genres,
    get_allowed_moods,
    load_catalog_records,
)
from src.recommender import recommend_songs
from src.spotify_enrich import has_credentials, search_track

EVAL_RESULTS_DIR = _REPO_ROOT / "eval" / "results"
EVAL_LATEST_CSV = EVAL_RESULTS_DIR / "latest.csv"
EVAL_SUMMARY_JSON = EVAL_RESULTS_DIR / "summary.json"

st.set_page_config(
    page_title="SoundFit — Conversational Music Recommender",
    page_icon=":notes:",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Loading catalog...")
def _load_resources():
    songs = load_catalog_records()
    return songs, get_allowed_genres(songs), get_allowed_moods()


@st.cache_resource(show_spinner=False)
def _warmup_once() -> None:
    warmup()


def _init_state() -> None:
    if "history" not in st.session_state:
        st.session_state.history = []
    if "last_turn" not in st.session_state:
        st.session_state.last_turn = None
    if "last_baseline_table" not in st.session_state:
        st.session_state.last_baseline_table = None


# ---------------------------------------------------------------------------
# Card rendering
# ---------------------------------------------------------------------------


def _render_feature_bars(card: Dict[str, Any]) -> None:
    cols = st.columns(4)
    with cols[0]:
        st.progress(min(1.0, float(card["energy"])), text=f"Energy {card['energy']:.2f}")
    with cols[1]:
        st.progress(min(1.0, float(card["valence"])), text=f"Valence {card['valence']:.2f}")
    with cols[2]:
        st.progress(
            min(1.0, float(card["danceability"])),
            text=f"Dance {card['danceability']:.2f}",
        )
    with cols[3]:
        st.caption(f"{float(card['tempo_bpm']):.0f} BPM")


def _render_card(card: Dict[str, Any], index: int) -> None:
    with st.container(border=True):
        title_col, badge_col = st.columns([3, 1])
        with title_col:
            st.markdown(f"**{index}. {card['title']}**")
            st.caption(f"by {card['artist']}")
        with badge_col:
            st.markdown(
                f"`{card['genre']}` `{card['mood']}`  \n"
                f"score: **{card['score']:.2f}**"
            )

        _render_feature_bars(card)

        if card.get("explanation"):
            st.markdown(f"_{card['explanation']}_")

        if has_credentials():
            spot = search_track(card["title"], card["artist"])
            if spot:
                enrich_cols = st.columns([1, 3])
                with enrich_cols[0]:
                    if spot.get("album_art_url"):
                        st.image(spot["album_art_url"], width=120)
                with enrich_cols[1]:
                    if spot.get("preview_url"):
                        st.audio(spot["preview_url"])
                    if spot.get("spotify_url"):
                        st.markdown(f"[Open in Spotify]({spot['spotify_url']})")


# ---------------------------------------------------------------------------
# Agent execution
# ---------------------------------------------------------------------------


def _agent_history() -> List[Dict[str, Any]]:
    """Pass only the conversation-relevant fields to run_agent."""
    return [
        {
            "role": h["role"],
            "content": h.get("content", ""),
            "parsed_intent": h.get("parsed_intent"),
        }
        for h in st.session_state.history
        if h.get("role") == "user"
    ]


def _baseline_keyword_prefs(
    query: str, allowed_genres: List[str], allowed_moods: List[str]
) -> Dict[str, Any]:
    q = query.lower()
    prefs: Dict[str, Any] = {}
    for g in allowed_genres:
        if g in q:
            prefs["genre"] = g
            break
    for m in allowed_moods:
        if m in q:
            prefs["mood"] = m
            break
    return prefs


def _run_baseline(
    query: str,
    songs: List[Dict[str, Any]],
    allowed_genres: List[str],
    allowed_moods: List[str],
    k: int = 5,
):
    prefs = _baseline_keyword_prefs(query, allowed_genres, allowed_moods)
    ranked = recommend_songs(prefs, songs, k=k)
    cards: List[Dict[str, Any]] = []
    for song, score, reasons in ranked:
        cards.append(
            {
                "song_id": int(song["id"]),
                "title": str(song["title"]),
                "artist": str(song["artist"]),
                "genre": str(song["genre"]),
                "mood": str(song["mood"]),
                "energy": float(song["energy"]),
                "valence": float(song["valence"]),
                "danceability": float(song["danceability"]),
                "acousticness": float(song["acousticness"]),
                "tempo_bpm": float(song["tempo_bpm"]),
                "score": float(score),
                "deterministic_reasons": reasons,
                "explanation": reasons or "Baseline match — no LLM in this mode.",
            }
        )
    return cards, prefs


# ---------------------------------------------------------------------------
# Sidebar tabs
# ---------------------------------------------------------------------------


def _render_trace_tab() -> None:
    st.caption(
        "What the agent extracted and how the deterministic scorer ranked the "
        "candidates on the most recent turn."
    )
    last = st.session_state.get("last_turn")
    last_baseline = st.session_state.get("last_baseline_table")

    if last is None and last_baseline is None:
        st.info("Type a request to see the agent's reasoning here.")
        return

    if last is not None:
        with st.expander("Parsed intent (LLM call #1 output)", expanded=True):
            st.json(last["parsed_intent"])
        with st.expander("Top candidates (deterministic scorer output)"):
            df = pd.DataFrame(last["cards"])[
                [
                    "song_id",
                    "title",
                    "artist",
                    "genre",
                    "mood",
                    "score",
                    "energy",
                    "valence",
                    "danceability",
                    "acousticness",
                    "tempo_bpm",
                ]
            ]
            st.dataframe(df, use_container_width=True, hide_index=True)
        if last.get("notes"):
            st.warning("\n".join(last["notes"]))
    elif last_baseline is not None:
        with st.expander("Baseline keyword extraction", expanded=True):
            st.json(last_baseline["prefs"])
        with st.expander("Top candidates"):
            st.dataframe(
                pd.DataFrame(last_baseline["cards"])[
                    [
                        "song_id",
                        "title",
                        "artist",
                        "genre",
                        "mood",
                        "score",
                        "energy",
                        "valence",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )


def _render_eval_tab() -> None:
    st.caption(
        "Live metrics from the YAML-driven eval harness. "
        "Regenerate with `pytest --ollama`."
    )
    if not EVAL_SUMMARY_JSON.exists():
        st.info(
            "No eval results yet.\n\n"
            "Run `pytest --ollama` (~5 min on CPU) to populate "
            "`eval/results/latest.csv` and `summary.json`."
        )
        return

    summary = json.loads(EVAL_SUMMARY_JSON.read_text())

    metric_cols = st.columns(3)
    with metric_cols[0]:
        st.metric("Pass rate", f"{summary['pass_rate']*100:.0f}%")
    with metric_cols[1]:
        st.metric("Cases", f"{summary['passed']}/{summary['total_cases']}")
    with metric_cols[2]:
        halluc = summary.get("hallucinated_id_count", 0)
        st.metric(
            "Hallucinated IDs",
            halluc,
            delta="violation" if halluc > 0 else "clean",
            delta_color="inverse" if halluc > 0 else "off",
        )

    if summary.get("by_type"):
        st.markdown("**By case type**")
        type_df = pd.DataFrame(
            [
                {"type": t, "passed": v["passed"], "total": v["total"], "rate": v["rate"]}
                for t, v in summary["by_type"].items()
            ]
        )
        type_df["rate"] = (type_df["rate"] * 100).round(0).astype(int).astype(str) + "%"
        st.dataframe(type_df, use_container_width=True, hide_index=True)

    if summary.get("by_category"):
        with st.expander("By category"):
            cat_df = pd.DataFrame(
                [
                    {"category": c, "passed": v["passed"], "total": v["total"], "rate": v["rate"]}
                    for c, v in summary["by_category"].items()
                ]
            )
            cat_df["rate"] = (cat_df["rate"] * 100).round(0).astype(int).astype(str) + "%"
            st.dataframe(cat_df, use_container_width=True, hide_index=True)

    if EVAL_LATEST_CSV.exists():
        df = pd.read_csv(EVAL_LATEST_CSV)
        failed = df[~df["passed"]]
        if not failed.empty:
            with st.expander(f"Failed cases ({len(failed)})"):
                st.dataframe(
                    failed[["case_id", "case_type", "query", "failed_assertions"]],
                    use_container_width=True,
                    hide_index=True,
                )


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------


def main() -> None:
    _init_state()
    songs, allowed_genres, allowed_moods = _load_resources()

    with st.sidebar:
        st.markdown("### SoundFit")
        st.caption(f"{len(songs)} tracks · {len(allowed_genres)} genres")

        baseline_only = st.toggle(
            "Baseline mode (no LLM)",
            value=False,
            help="Skip Ollama and use the rule-based scorer directly. Useful "
            "to A/B against the agent.",
        )

        if st.button("Reset chat", use_container_width=True):
            st.session_state.history = []
            st.session_state.last_turn = None
            st.session_state.last_baseline_table = None
            st.rerun()

        st.divider()
        if has_credentials():
            st.success("Spotify enrichment: ON")
        else:
            st.caption(
                "Spotify enrichment: off.  \nSet `SPOTIFY_CLIENT_ID` and "
                "`SPOTIFY_CLIENT_SECRET` to add album art and 30s previews."
            )

        st.divider()
        trace_tab, eval_tab = st.tabs(["Trace", "Eval"])
        with trace_tab:
            _render_trace_tab()
        with eval_tab:
            _render_eval_tab()

    st.title("SoundFit")
    st.caption(
        "A local-first agentic music recommender. Tell it what you're in the "
        "mood for — it parses intent with `qwen2.5:3b` (Ollama), ranks with a "
        "deterministic feature scorer, and writes grounded explanations."
    )

    if not baseline_only:
        _warmup_once()

    for msg in st.session_state.history:
        with st.chat_message(msg["role"]):
            if msg["role"] == "user":
                st.write(msg["content"])
            else:
                if msg.get("notes"):
                    for n in msg["notes"]:
                        st.info(n)
                cards = msg.get("cards", [])
                if not cards:
                    st.write("No matches.")
                for i, card in enumerate(cards, 1):
                    _render_card(card, i)

    placeholder = (
        "e.g. 'aux rap songs', 'late night studying', 'songs by Drake', "
        "'more upbeat though'"
    )
    if query := st.chat_input(placeholder):
        with st.chat_message("user"):
            st.write(query)

        with st.chat_message("assistant"):
            if baseline_only:
                with st.spinner("Scoring..."):
                    cards, prefs = _run_baseline(
                        query, songs, allowed_genres, allowed_moods
                    )
                st.session_state.last_baseline_table = {"prefs": prefs, "cards": cards}
                st.session_state.last_turn = None
                st.session_state.history.append({"role": "user", "content": query})
                st.session_state.history.append(
                    {"role": "assistant", "cards": cards, "notes": []}
                )
                for i, card in enumerate(cards, 1):
                    _render_card(card, i)
            else:
                with st.spinner("The agent is thinking..."):
                    try:
                        turn = run_agent(
                            query,
                            songs,
                            _agent_history(),
                            allowed_genres,
                            allowed_moods,
                        )
                    except OllamaUnavailableError as exc:
                        st.error(
                            f"Ollama unavailable — {exc}. "
                            f"Toggle Baseline mode in the sidebar to continue."
                        )
                        st.stop()

                cards = [c.model_dump() for c in turn.cards]
                st.session_state.last_turn = {
                    "parsed_intent": turn.parsed_intent.model_dump(),
                    "cards": cards,
                    "notes": turn.notes,
                    "used_baseline_fallback": turn.used_baseline_fallback,
                }
                st.session_state.last_baseline_table = None

                st.session_state.history.append(
                    {
                        "role": "user",
                        "content": query,
                        "parsed_intent": turn.parsed_intent.model_dump(),
                    }
                )
                st.session_state.history.append(
                    {"role": "assistant", "cards": cards, "notes": turn.notes}
                )

                if turn.used_baseline_fallback:
                    st.warning("Ollama dropped — these are baseline rule-based picks.")
                for note in turn.notes:
                    st.info(note)
                if not cards:
                    st.write("No matches.")
                for i, card in enumerate(cards, 1):
                    _render_card(card, i)


if __name__ == "__main__":
    main()
