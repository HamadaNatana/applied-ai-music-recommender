"""Unit tests for src/schemas.py — Pydantic intent validation."""

import pytest
from pydantic import ValidationError

from src.schemas import IntentSchema


def test_intent_validates_basic_input():
    intent = IntentSchema(genre="pop", mood="happy", energy=0.8, k=5)
    assert intent.genre == "pop"
    assert intent.mood == "happy"
    assert intent.energy == 0.8
    assert intent.k == 5


def test_intent_lowercases_genre_and_mood():
    intent = IntentSchema(genre="POP", mood="Happy")
    assert intent.genre == "pop"
    assert intent.mood == "happy"


def test_intent_treats_null_strings_as_none():
    for sentinel in ("null", "None", "n/a", ""):
        intent = IntentSchema(genre=sentinel)
        assert intent.genre is None


def test_intent_rejects_out_of_range_energy():
    with pytest.raises(ValidationError):
        IntentSchema(energy=1.5)
    with pytest.raises(ValidationError):
        IntentSchema(energy=-0.1)


def test_intent_defaults_are_safe():
    intent = IntentSchema()
    assert intent.genre is None
    assert intent.mood is None
    assert intent.k == 5
    assert intent.refinement_of_previous is False
    assert intent.exclude_song_ids == []


def test_to_user_prefs_strips_none_fields():
    intent = IntentSchema(genre="rock", energy=0.9)
    prefs = intent.to_user_prefs()
    assert prefs == {"genre": "rock", "energy": 0.9}
    assert "mood" not in prefs


def test_to_user_prefs_empty_when_all_none():
    intent = IntentSchema()
    assert intent.to_user_prefs() == {}


def test_intent_strips_whitespace():
    intent = IntentSchema(genre="  POP  ")
    assert intent.genre == "pop"
