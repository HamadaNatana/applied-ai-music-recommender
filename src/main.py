"""CLI runner for the SoundFit music recommender.

Modes:
  python -m src.main                # interactive chat with the agent (default)
  python -m src.main --demo         # legacy 4-profile baseline demo
  python -m src.main --baseline     # interactive REPL using only the rule-based scorer
"""

from __future__ import annotations

import argparse
from typing import Any, Dict, List

from src.recommender import load_songs, recommend_songs


def _format_card(card) -> str:
    return (
        f"  • {card.title} — {card.artist} [{card.genre}/{card.mood}] \n"
        f"    score = {card.score:.2f}  energy = {card.energy:.2f}  "
        f"valence = {card.valence:.2f}  tempo = {card.tempo_bpm:.0f} bpm \n"
        f"    {card.explanation}"
    )


def run_demo() -> None:
    """The original 4-profile baseline demo, preserved for backward compat."""
    songs = load_songs("data/songs.csv")
    print(f"Loaded {len(songs)} songs (legacy 10-track catalog).")
    profiles: List[Dict[str, Any]] = [
        {"genre": "pop", "mood": "happy", "energy": 0.8},
        {"genre": "lofi", "mood": "chill", "energy": 0.4},
        {"genre": "rock", "mood": "angry", "energy": 0.2},
        {"genre": "jazz", "mood": "intense", "energy": 0.8},
    ]
    for i, prefs in enumerate(profiles, 1):
        print(f"\nTop recommendations for User {i}: {prefs}")
        for song, score, expl in recommend_songs(prefs, songs, k=5):
            print(f"  {song['title']} - Score: {score:.2f}")
            print(f"  Because: {expl}\n")


def _baseline_keyword_prefs(query: str, allowed_genres, allowed_moods) -> Dict[str, Any]:
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


def run_repl(use_agent: bool = True) -> None:
    from src.data import (
        get_allowed_genres,
        get_allowed_moods,
        load_catalog_records,
    )

    songs = load_catalog_records()
    allowed_genres = get_allowed_genres(songs)
    allowed_moods = get_allowed_moods()
    print(f"Loaded {len(songs)} songs across {len(allowed_genres)} genre families.")

    history: List[Dict[str, Any]] = []

    if use_agent:
        from src.agent import OllamaUnavailableError, run_agent, warmup

        print("Warming up Ollama (first call is slow)...")
        warmup()
        print("Ready. Type a request, or 'exit' to quit.\n")
    else:
        print("Baseline mode (rule-based, no LLM).\n")

    while True:
        try:
            query = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not query:
            continue
        if query.lower() in {"exit", "quit", ":q"}:
            break
        if query.lower() in {"reset", "clear", ":reset"}:
            history.clear()
            print("[chat history cleared]\n")
            continue

        if use_agent:
            turn = run_agent(query, songs, history, allowed_genres, allowed_moods)
            if turn.used_baseline_fallback:
                print("[!] Ollama unavailable — falling back to rule-based recommendations.")
            for note in turn.notes:
                print(f"[note] {note}")
            for card in turn.cards:
                print(_format_card(card))
            history.append(
                {
                    "role": "user",
                    "content": query,
                    "parsed_intent": turn.parsed_intent.model_dump(),
                }
            )
            print()
        else:
            prefs = _baseline_keyword_prefs(query, allowed_genres, allowed_moods)
            top = recommend_songs(prefs, songs, k=5)
            for song, score, expl in top:
                print(
                    f"  • {song['title']} — {song['artist']} "
                    f"[{song['genre']}/{song['mood']}]  score={score:.2f}\n"
                    f"    {expl or 'no preference signal extracted from query'}"
                )
            print()


def main() -> None:
    parser = argparse.ArgumentParser(description="SoundFit music recommender CLI.")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run the legacy 4-profile baseline demo on the original 10-track catalog.",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Interactive REPL using only the rule-based scorer (no LLM).",
    )
    args = parser.parse_args()

    if args.demo:
        run_demo()
    elif args.baseline:
        run_repl(use_agent=False)
    else:
        run_repl(use_agent=True)


if __name__ == "__main__":
    main()
