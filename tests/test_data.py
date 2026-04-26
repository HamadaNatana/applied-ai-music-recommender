"""Unit tests for src/data.py — catalog loading, genre normalization, mood
derivation, and integration with the existing scorer.

These tests deliberately avoid the HuggingFace download path so they run
fast and offline. The full HF pipeline is exercised manually / in CI."""

import pandas as pd
import pytest

from src.data import (
    ALL_GENRE_FAMILIES,
    ALL_MOODS,
    derive_mood,
    get_allowed_genres,
    get_allowed_moods,
    normalize_genre,
    _load_sample_csv,
)
from src.recommender import recommend_songs, score_song


def test_normalize_genre_maps_known_genres():
    assert normalize_genre("pop") == "pop"
    assert normalize_genre("indie pop") == "pop"
    assert normalize_genre("alt-rock") == "rock"
    assert normalize_genre("trap") == "hip-hop"
    assert normalize_genre("deep-house") == "electronic"
    assert normalize_genre("lofi") == "electronic"


def test_normalize_genre_unknown_returns_other():
    assert normalize_genre("polka") == "other"
    assert normalize_genre("") == "other"
    assert normalize_genre(None) == "other"


def test_normalize_genre_handles_compound_strings():
    # "japanese-trap" should land in hip-hop via the "trap" suffix
    assert normalize_genre("japanese-trap") == "hip-hop"


def test_derive_mood_high_energy_high_valence_is_happy():
    row = pd.Series({"valence": 0.85, "energy": 0.80, "acousticness": 0.1, "instrumentalness": 0.0})
    assert derive_mood(row) == "happy"


def test_derive_mood_low_energy_low_valence_is_moody():
    row = pd.Series({"valence": 0.2, "energy": 0.3, "acousticness": 0.4, "instrumentalness": 0.0})
    assert derive_mood(row) == "moody"


def test_derive_mood_high_energy_low_valence_is_intense():
    row = pd.Series({"valence": 0.25, "energy": 0.85, "acousticness": 0.0, "instrumentalness": 0.0})
    assert derive_mood(row) == "intense"


def test_derive_mood_high_instrumental_low_energy_is_focused():
    row = pd.Series({"valence": 0.5, "energy": 0.25, "acousticness": 0.9, "instrumentalness": 0.85})
    assert derive_mood(row) == "focused"


def test_load_sample_csv_returns_at_least_50_records():
    df = _load_sample_csv()
    records = df.to_dict("records")
    assert len(records) >= 50


def test_sample_records_have_required_schema():
    required = {"id", "title", "artist", "genre", "mood", "energy",
                "tempo_bpm", "valence", "danceability", "acousticness"}
    records = _load_sample_csv().to_dict("records")
    for rec in records:
        missing = required - set(rec.keys())
        assert not missing, f"Record {rec.get('id')} missing keys: {missing}"


def test_sample_records_use_normalized_genre_families():
    records = _load_sample_csv().to_dict("records")
    found = {rec["genre"] for rec in records}
    assert found.issubset(set(ALL_GENRE_FAMILIES)), (
        f"Sample contains genres outside the family list: {found - set(ALL_GENRE_FAMILIES)}"
    )


def test_sample_records_use_canonical_moods():
    records = _load_sample_csv().to_dict("records")
    found = {rec["mood"] for rec in records}
    assert found.issubset(set(ALL_MOODS)), (
        f"Sample contains moods outside canonical set: {found - set(ALL_MOODS)}"
    )


def test_sample_records_feed_through_scorer_unchanged():
    """Integration check: existing dict-based scorer in recommender.py
    accepts the new sample catalog records without modification."""
    records = _load_sample_csv().to_dict("records")
    prefs = {"genre": "pop", "mood": "happy", "energy": 0.8}
    score, reasons = score_song(prefs, records[1])  # "Shape of You" – pop/happy
    assert score > 0
    assert any("Genre" in r for r in reasons)


def test_recommend_songs_returns_top_k_from_sample():
    records = _load_sample_csv().to_dict("records")
    prefs = {"genre": "pop", "mood": "happy", "energy": 0.8}
    top = recommend_songs(prefs, records, k=5)
    assert len(top) == 5
    # First result should be a pop song (genre match dominates)
    first_song, _, _ = top[0]
    assert first_song["genre"] == "pop"


def test_get_allowed_genres_reflects_loaded_catalog():
    records = _load_sample_csv().to_dict("records")
    allowed = get_allowed_genres(records)
    assert "pop" in allowed
    assert "rock" in allowed
    assert "jazz" in allowed
    assert allowed == sorted(allowed)


def test_get_allowed_moods_returns_canonical_list():
    moods = get_allowed_moods()
    assert set(moods) == set(ALL_MOODS)
