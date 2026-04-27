"""Evaluation harness for the SoundFit agent.

Loads YAML case files, runs each case against the live agent (requires
Ollama), checks property-based assertions, and writes a wide-format CSV
to eval/results/latest.csv plus a timestamped copy. The Streamlit Eval
tab and the pytest entry point both read the CSV via pandas.

Property assertions supported (golden + refinement cases):
  top1_genre_in: [list]            -- top result's genre must be in list
  top1_genre_eq: str
  top1_mood_in: [list]
  top1_mood_eq: str
  top1_energy_min / top1_energy_max: float
  top1_valence_min / top1_valence_max: float
  top1_acousticness_min / top1_acousticness_max: float
  top1_artist_lower_contains: str
  top1_artist_lower_excludes: str
  topk_avg_energy_min / topk_avg_energy_max: float
  topk_avg_valence_min / topk_avg_valence_max: float
  topk_min_count: int              -- minimum number of cards returned
  delta_topk_avg_energy_min/max: float   -- (refinement only) compared to previous turn

Behavioral cases (intent parser only):
  genre / mood: str|null            -- exact match (or strict null)
  genre_in / mood_in: [list]        -- genre must be in list
  artist: null                      -- must be null
  artist_lower_contains: str        -- artist name must contain (case-insensitive)
  energy_min / energy_max: float
  valence_min / valence_max: float
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = REPO_ROOT / "eval"
RESULTS_DIR = EVAL_DIR / "results"
LATEST_CSV = RESULTS_DIR / "latest.csv"
SUMMARY_JSON = RESULTS_DIR / "summary.json"


@dataclass
class CaseResult:
    case_id: str
    case_type: str  # "golden" | "behavioral" | "refinement"
    category: Optional[str]
    query: str
    passed: bool
    failed_assertions: List[str] = field(default_factory=list)
    hallucinated_ids: List[int] = field(default_factory=list)
    top1_id: Optional[int] = None
    top1_artist: Optional[str] = None
    top1_genre: Optional[str] = None
    top1_mood: Optional[str] = None
    top1_energy: Optional[float] = None
    topk_avg_energy: Optional[float] = None
    error: Optional[str] = None


def load_yaml_cases(path: Path) -> List[Dict[str, Any]]:
    """Read a YAML case file. Returns an empty list if the file is missing."""
    if not path.exists():
        return []
    with open(path, "r") as f:
        data = yaml.safe_load(f) or []
    if not isinstance(data, list):
        raise ValueError(f"{path} must be a YAML list of cases")
    return data


# ---------------------------------------------------------------------------
# Property assertions
# ---------------------------------------------------------------------------


def _check_top1(top1: Dict[str, Any], expected: Dict[str, Any]) -> List[str]:
    failures: List[str] = []
    if "top1_genre_in" in expected and top1.get("genre") not in expected["top1_genre_in"]:
        failures.append(f"top1_genre={top1.get('genre')} not in {expected['top1_genre_in']}")
    if "top1_genre_eq" in expected and top1.get("genre") != expected["top1_genre_eq"]:
        failures.append(f"top1_genre={top1.get('genre')} != {expected['top1_genre_eq']}")
    if "top1_mood_in" in expected and top1.get("mood") not in expected["top1_mood_in"]:
        failures.append(f"top1_mood={top1.get('mood')} not in {expected['top1_mood_in']}")
    if "top1_mood_eq" in expected and top1.get("mood") != expected["top1_mood_eq"]:
        failures.append(f"top1_mood={top1.get('mood')} != {expected['top1_mood_eq']}")
    for feat in ("energy", "valence", "danceability", "acousticness"):
        min_key = f"top1_{feat}_min"
        max_key = f"top1_{feat}_max"
        actual = top1.get(feat)
        if actual is None:
            continue
        if min_key in expected and float(actual) < float(expected[min_key]):
            failures.append(f"top1_{feat}={actual:.2f} < min {expected[min_key]}")
        if max_key in expected and float(actual) > float(expected[max_key]):
            failures.append(f"top1_{feat}={actual:.2f} > max {expected[max_key]}")
    if "top1_artist_lower_contains" in expected:
        artist_lc = str(top1.get("artist", "")).lower()
        needle = str(expected["top1_artist_lower_contains"]).lower()
        if needle not in artist_lc:
            failures.append(f"top1_artist='{top1.get('artist')}' missing '{needle}'")
    if "top1_artist_lower_excludes" in expected:
        artist_lc = str(top1.get("artist", "")).lower()
        needle = str(expected["top1_artist_lower_excludes"]).lower()
        if needle in artist_lc:
            failures.append(f"top1_artist='{top1.get('artist')}' should not contain '{needle}'")
    return failures


def _check_topk(cards: List[Dict[str, Any]], expected: Dict[str, Any]) -> List[str]:
    failures: List[str] = []
    if "topk_min_count" in expected and len(cards) < int(expected["topk_min_count"]):
        failures.append(f"topk_count={len(cards)} < {expected['topk_min_count']}")
    if not cards:
        return failures
    avg_energy = sum(float(c.get("energy", 0.0)) for c in cards) / len(cards)
    avg_valence = sum(float(c.get("valence", 0.0)) for c in cards) / len(cards)
    if "topk_avg_energy_min" in expected and avg_energy < float(expected["topk_avg_energy_min"]):
        failures.append(f"topk_avg_energy={avg_energy:.2f} < {expected['topk_avg_energy_min']}")
    if "topk_avg_energy_max" in expected and avg_energy > float(expected["topk_avg_energy_max"]):
        failures.append(f"topk_avg_energy={avg_energy:.2f} > {expected['topk_avg_energy_max']}")
    if "topk_avg_valence_min" in expected and avg_valence < float(expected["topk_avg_valence_min"]):
        failures.append(f"topk_avg_valence={avg_valence:.2f} < {expected['topk_avg_valence_min']}")
    if "topk_avg_valence_max" in expected and avg_valence > float(expected["topk_avg_valence_max"]):
        failures.append(f"topk_avg_valence={avg_valence:.2f} > {expected['topk_avg_valence_max']}")
    return failures


def _check_intent(intent_dump: Dict[str, Any], expected: Dict[str, Any]) -> List[str]:
    """For behavioral cases — assert against parsed intent fields."""
    failures: List[str] = []
    for key in ("genre", "mood"):
        if key in expected:
            exp = expected[key]
            actual = intent_dump.get(key)
            if exp is None and actual is not None:
                failures.append(f"{key} expected null, got '{actual}'")
            elif exp is not None and actual != exp:
                failures.append(f"{key} expected '{exp}', got '{actual}'")
    for key in ("genre_in", "mood_in"):
        base = key[:-3]
        if key in expected:
            actual = intent_dump.get(base)
            if actual not in expected[key]:
                failures.append(f"{base}='{actual}' not in {expected[key]}")
    if "artist" in expected:
        if expected["artist"] is None and intent_dump.get("artist") is not None:
            failures.append(f"artist expected null, got '{intent_dump['artist']}'")
    if "artist_lower_contains" in expected:
        artist_lc = str(intent_dump.get("artist") or "").lower()
        needle = str(expected["artist_lower_contains"]).lower()
        if needle not in artist_lc:
            failures.append(f"artist='{intent_dump.get('artist')}' missing '{needle}'")
    for feat in ("energy", "valence", "danceability", "acousticness"):
        actual = intent_dump.get(feat)
        if actual is None:
            if f"{feat}_min" in expected or f"{feat}_max" in expected:
                failures.append(f"{feat} is null but had bound assertion")
            continue
        if f"{feat}_min" in expected and float(actual) < float(expected[f"{feat}_min"]):
            failures.append(f"{feat}={actual:.2f} < {expected[f'{feat}_min']}")
        if f"{feat}_max" in expected and float(actual) > float(expected[f"{feat}_max"]):
            failures.append(f"{feat}={actual:.2f} > {expected[f'{feat}_max']}")
    return failures


# ---------------------------------------------------------------------------
# Case runners
# ---------------------------------------------------------------------------


def _cards_to_dicts(cards) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for c in cards:
        d = c.model_dump() if hasattr(c, "model_dump") else dict(c)
        d["id"] = d.get("song_id", d.get("id"))
        out.append(d)
    return out


def run_golden_case(
    case: Dict[str, Any],
    songs: List[Dict[str, Any]],
    allowed_genres: List[str],
    allowed_moods: List[str],
) -> CaseResult:
    from src.agent import run_agent

    catalog_ids = {int(s["id"]) for s in songs}
    try:
        turn = run_agent(case["query"], songs, [], allowed_genres, allowed_moods)
    except Exception as exc:  # noqa: BLE001
        return CaseResult(
            case_id=case["id"], case_type="golden",
            category=case.get("category"), query=case["query"],
            passed=False, error=str(exc),
        )

    cards = _cards_to_dicts(turn.cards)
    hallucinated = [int(c["id"]) for c in cards if int(c["id"]) not in catalog_ids]
    expected = case.get("expect", {})
    failures: List[str] = []
    if cards:
        failures.extend(_check_top1(cards[0], expected))
    failures.extend(_check_topk(cards, expected))

    top1 = cards[0] if cards else {}
    avg_energy = (
        sum(float(c.get("energy", 0.0)) for c in cards) / len(cards) if cards else None
    )
    return CaseResult(
        case_id=case["id"], case_type="golden",
        category=case.get("category"), query=case["query"],
        passed=(not failures and not hallucinated),
        failed_assertions=failures, hallucinated_ids=hallucinated,
        top1_id=int(top1["id"]) if top1 else None,
        top1_artist=top1.get("artist"),
        top1_genre=top1.get("genre"),
        top1_mood=top1.get("mood"),
        top1_energy=float(top1["energy"]) if top1 else None,
        topk_avg_energy=avg_energy,
    )


def run_behavioral_case(
    case: Dict[str, Any],
    allowed_genres: List[str],
    allowed_moods: List[str],
) -> CaseResult:
    from src.agent import parse_intent

    try:
        intent = parse_intent(case["query"], [], allowed_genres, allowed_moods)
    except Exception as exc:  # noqa: BLE001
        return CaseResult(
            case_id=case["id"], case_type="behavioral",
            category=None, query=case["query"], passed=False, error=str(exc),
        )

    failures = _check_intent(intent.model_dump(), case.get("expect", {}))
    return CaseResult(
        case_id=case["id"], case_type="behavioral",
        category=None, query=case["query"],
        passed=not failures, failed_assertions=failures,
        top1_artist=intent.artist, top1_genre=intent.genre, top1_mood=intent.mood,
    )


def run_refinement_case(
    case: Dict[str, Any],
    songs: List[Dict[str, Any]],
    allowed_genres: List[str],
    allowed_moods: List[str],
) -> List[CaseResult]:
    from src.agent import run_agent

    catalog_ids = {int(s["id"]) for s in songs}
    history: List[Dict[str, Any]] = []
    results: List[CaseResult] = []
    prev_avg_energy: Optional[float] = None

    for i, turn_spec in enumerate(case.get("turns", [])):
        try:
            turn = run_agent(turn_spec["query"], songs, history, allowed_genres, allowed_moods)
        except Exception as exc:  # noqa: BLE001
            results.append(CaseResult(
                case_id=f"{case['id']}_turn{i+1}", case_type="refinement",
                category=case.get("category", "refinement"),
                query=turn_spec["query"], passed=False, error=str(exc),
            ))
            continue

        cards = _cards_to_dicts(turn.cards)
        hallucinated = [int(c["id"]) for c in cards if int(c["id"]) not in catalog_ids]
        expected = turn_spec.get("expect", {})
        failures: List[str] = []
        if cards:
            failures.extend(_check_top1(cards[0], expected))
        failures.extend(_check_topk(cards, expected))

        avg_energy = (
            sum(float(c.get("energy", 0.0)) for c in cards) / len(cards) if cards else None
        )
        if prev_avg_energy is not None and avg_energy is not None:
            delta = avg_energy - prev_avg_energy
            if "delta_topk_avg_energy_min" in expected and delta < float(expected["delta_topk_avg_energy_min"]):
                failures.append(
                    f"delta_topk_avg_energy={delta:+.2f} < min {expected['delta_topk_avg_energy_min']}"
                )
            if "delta_topk_avg_energy_max" in expected and delta > float(expected["delta_topk_avg_energy_max"]):
                failures.append(
                    f"delta_topk_avg_energy={delta:+.2f} > max {expected['delta_topk_avg_energy_max']}"
                )

        top1 = cards[0] if cards else {}
        results.append(CaseResult(
            case_id=f"{case['id']}_turn{i+1}", case_type="refinement",
            category=case.get("category", "refinement"),
            query=turn_spec["query"],
            passed=(not failures and not hallucinated),
            failed_assertions=failures, hallucinated_ids=hallucinated,
            top1_id=int(top1["id"]) if top1 else None,
            top1_artist=top1.get("artist"),
            top1_genre=top1.get("genre"),
            top1_mood=top1.get("mood"),
            top1_energy=float(top1["energy"]) if top1 else None,
            topk_avg_energy=avg_energy,
        ))
        prev_avg_energy = avg_energy
        history.append({
            "role": "user",
            "content": turn_spec["query"],
            "parsed_intent": turn.parsed_intent.model_dump(),
        })

    return results


def run_all_cases(
    songs: List[Dict[str, Any]],
    allowed_genres: List[str],
    allowed_moods: List[str],
    eval_dir: Path = EVAL_DIR,
) -> List[CaseResult]:
    """Run every case across all three YAML files and return a flat list."""
    results: List[CaseResult] = []
    for case in load_yaml_cases(eval_dir / "behavioral_cases.yaml"):
        results.append(run_behavioral_case(case, allowed_genres, allowed_moods))
    for case in load_yaml_cases(eval_dir / "golden_cases.yaml"):
        results.append(run_golden_case(case, songs, allowed_genres, allowed_moods))
    for case in load_yaml_cases(eval_dir / "refinement_cases.yaml"):
        results.extend(run_refinement_case(case, songs, allowed_genres, allowed_moods))
    return results


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def write_results(
    results: List[CaseResult],
    results_dir: Path = RESULTS_DIR,
) -> Tuple[pd.DataFrame, Path, Path]:
    """Write a wide-format CSV with one row per case. Also writes a
    timestamped copy alongside latest.csv. Returns (df, latest_path, ts_path)."""
    rows = [asdict(r) for r in results]
    df = pd.DataFrame(rows)
    if not df.empty:
        df["failed_assertions"] = df["failed_assertions"].apply(
            lambda lst: " | ".join(lst) if isinstance(lst, list) else str(lst)
        )
        df["hallucinated_ids"] = df["hallucinated_ids"].apply(
            lambda lst: ",".join(str(x) for x in lst) if isinstance(lst, list) else str(lst)
        )
    results_dir.mkdir(parents=True, exist_ok=True)
    latest = results_dir / "latest.csv"
    df.to_csv(latest, index=False)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamped = results_dir / f"{ts}.csv"
    df.to_csv(timestamped, index=False)
    return df, latest, timestamped


def summarize(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute headline metrics for the Streamlit Eval tab and pytest output."""
    if df.empty:
        return {"total_cases": 0}

    halluc_count = int(
        df["hallucinated_ids"].apply(lambda s: len(str(s).split(",")) if str(s) else 0).sum()
    ) if "hallucinated_ids" in df.columns else 0

    summary: Dict[str, Any] = {
        "total_cases": int(len(df)),
        "passed": int(df["passed"].sum()),
        "pass_rate": float(df["passed"].mean()),
        "hallucinated_id_count": halluc_count,
    }
    if "case_type" in df.columns:
        by_type = df.groupby("case_type")["passed"].agg(["sum", "count", "mean"])
        summary["by_type"] = {
            t: {"passed": int(row["sum"]), "total": int(row["count"]), "rate": float(row["mean"])}
            for t, row in by_type.iterrows()
        }
    if "category" in df.columns and df["category"].notna().any():
        by_cat = df[df["category"].notna()].groupby("category")["passed"].agg(["sum", "count", "mean"])
        summary["by_category"] = {
            c: {"passed": int(row["sum"]), "total": int(row["count"]), "rate": float(row["mean"])}
            for c, row in by_cat.iterrows()
        }
    summary["error_count"] = int(df["error"].notna().sum()) if "error" in df.columns else 0
    return summary
