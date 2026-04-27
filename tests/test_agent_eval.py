"""End-to-end evaluation harness driven by YAML cases.

Marked `@pytest.mark.ollama` — skipped by default. Run with:
    pytest --ollama -v

These tests:
- run every behavioral / golden / refinement case in eval/*.yaml
- write results to eval/results/latest.csv (read by Streamlit Eval tab)
- hard-fail if the agent ever returns a song ID not in the catalog
- soft-fail if the overall pass rate drops below thresholds (per type)

Property assertions are documented in eval/harness.py — they're
intentionally loose since LLM output is nondeterministic at temp=0 only
to within a tolerance."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.harness import (
    LATEST_CSV,
    SUMMARY_JSON,
    run_all_cases,
    summarize,
    write_results,
)
from src.data import (
    get_allowed_genres,
    get_allowed_moods,
    load_catalog_records,
)


@pytest.fixture(scope="module")
def catalog():
    songs = load_catalog_records()
    return songs, get_allowed_genres(songs), get_allowed_moods()


@pytest.fixture(scope="module")
def eval_results(catalog):
    """Run the full eval suite once per pytest module. Subsequent test
    functions consume the same results to avoid re-running the LLM."""
    songs, ag, am = catalog
    results = run_all_cases(songs, ag, am)
    df, latest, _ = write_results(results)
    summary = summarize(df)
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2))
    return df, summary


@pytest.mark.ollama
def test_no_hallucinated_song_ids(eval_results):
    """HARD GUARANTEE: the agent must never surface a song ID that isn't
    in the catalog. If this ever fails, the integrity contract is broken."""
    df, summary = eval_results
    assert summary["hallucinated_id_count"] == 0, (
        f"Agent returned songs not in catalog. See {LATEST_CSV}.\n"
        f"{df[df['hallucinated_ids'] != ''][['case_id', 'hallucinated_ids']].to_string()}"
    )


@pytest.mark.ollama
def test_no_runtime_errors(eval_results):
    df, summary = eval_results
    if summary.get("error_count", 0) > 0:
        errored = df[df["error"].notna()][["case_id", "query", "error"]].to_string()
        pytest.fail(f"Cases errored at runtime:\n{errored}")


@pytest.mark.ollama
def test_behavioral_pass_rate(eval_results):
    """Intent extraction should be solid — soft floor 70% pass rate.
    Behavioral cases test the LLM's ability to map clear queries to
    structured intent (e.g. 'I love pop' -> genre=pop)."""
    df, summary = eval_results
    by_type = summary.get("by_type", {})
    if "behavioral" not in by_type:
        pytest.skip("no behavioral cases ran")
    rate = by_type["behavioral"]["rate"]
    if rate < 0.7:
        failed = df[(df["case_type"] == "behavioral") & (~df["passed"])][
            ["case_id", "query", "failed_assertions"]
        ].to_string()
        pytest.fail(f"behavioral pass rate {rate:.0%} < 70%. Failures:\n{failed}")


@pytest.mark.ollama
def test_golden_pass_rate(eval_results):
    """End-to-end pass rate floor 60%. Property assertions are loose;
    golden cases mostly verify the scorer surfaces sensible tracks given
    the LLM-extracted intent."""
    df, summary = eval_results
    by_type = summary.get("by_type", {})
    if "golden" not in by_type:
        pytest.skip("no golden cases ran")
    rate = by_type["golden"]["rate"]
    if rate < 0.6:
        failed = df[(df["case_type"] == "golden") & (~df["passed"])][
            ["case_id", "query", "failed_assertions"]
        ].to_string()
        pytest.fail(f"golden pass rate {rate:.0%} < 60%. Failures:\n{failed}")


@pytest.mark.ollama
def test_refinement_pass_rate(eval_results):
    """Multi-turn refinements verify chat-history integration. Loose
    threshold — qwen2.5:3b is a small model and refinement detection is
    inherently noisier than single-shot extraction."""
    df, summary = eval_results
    by_type = summary.get("by_type", {})
    if "refinement" not in by_type:
        pytest.skip("no refinement cases ran")
    rate = by_type["refinement"]["rate"]
    if rate < 0.5:
        failed = df[(df["case_type"] == "refinement") & (~df["passed"])][
            ["case_id", "query", "failed_assertions"]
        ].to_string()
        pytest.fail(f"refinement pass rate {rate:.0%} < 50%. Failures:\n{failed}")


@pytest.mark.ollama
def test_summary_written_to_disk(eval_results):
    """CSV + JSON outputs exist and are readable — the Streamlit Eval tab
    depends on these files."""
    df, _ = eval_results
    assert LATEST_CSV.exists(), f"{LATEST_CSV} was not written"
    assert SUMMARY_JSON.exists(), f"{SUMMARY_JSON} was not written"
    assert len(df) > 0, "no eval cases ran"


# --- Harness unit tests (no Ollama needed) -----------------------------------


def test_load_yaml_cases_handles_missing_file(tmp_path):
    from eval.harness import load_yaml_cases

    missing = tmp_path / "nonexistent.yaml"
    assert load_yaml_cases(missing) == []


def test_load_yaml_cases_rejects_non_list(tmp_path):
    from eval.harness import load_yaml_cases

    bad = tmp_path / "bad.yaml"
    bad.write_text("not_a_list: true\n")
    with pytest.raises(ValueError):
        load_yaml_cases(bad)


def test_check_top1_genre_in_passes():
    from eval.harness import _check_top1
    failures = _check_top1({"genre": "pop", "energy": 0.8}, {"top1_genre_in": ["pop", "rock"]})
    assert failures == []


def test_check_top1_genre_in_fails():
    from eval.harness import _check_top1
    failures = _check_top1({"genre": "jazz"}, {"top1_genre_in": ["pop", "rock"]})
    assert any("genre" in f for f in failures)


def test_check_top1_energy_min_fails():
    from eval.harness import _check_top1
    failures = _check_top1({"energy": 0.3}, {"top1_energy_min": 0.7})
    assert any("energy" in f for f in failures)


def test_check_topk_avg_energy_min_passes():
    from eval.harness import _check_topk
    cards = [{"energy": 0.8}, {"energy": 0.85}, {"energy": 0.9}]
    failures = _check_topk(cards, {"topk_avg_energy_min": 0.7})
    assert failures == []


def test_check_intent_null_genre_assertion():
    from eval.harness import _check_intent
    failures = _check_intent({"genre": "pop"}, {"genre": None})
    assert any("null" in f for f in failures)


def test_check_intent_null_genre_passes_when_actually_null():
    from eval.harness import _check_intent
    failures = _check_intent({"genre": None}, {"genre": None})
    assert failures == []
