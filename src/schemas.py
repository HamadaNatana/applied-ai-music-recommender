"""Pydantic models for agent I/O. The intent schema is what the LLM is
asked to fill, validated on every turn. The recommendation card is what the
agent returns to the UI."""

from typing import List, Optional, Dict
from pydantic import BaseModel, Field, field_validator


class IntentSchema(BaseModel):
    genre: Optional[str] = None
    mood: Optional[str] = None
    artist: Optional[str] = None
    energy: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    valence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    danceability: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    acousticness: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    k: int = Field(default=5, ge=1, le=20)
    refinement_of_previous: bool = False
    exclude_song_ids: List[int] = Field(default_factory=list)

    @field_validator("genre", "mood", mode="before")
    @classmethod
    def lowercase_strings(cls, v):
        if isinstance(v, str):
            v = v.strip().lower()
            if v in ("", "null", "none", "n/a"):
                return None
        return v

    @field_validator("artist", mode="before")
    @classmethod
    def normalize_artist(cls, v):
        if isinstance(v, str):
            v = v.strip()
            if v.lower() in ("", "null", "none", "n/a"):
                return None
        return v

    def to_user_prefs(self) -> Dict:
        prefs: Dict = {}
        for key in ("genre", "mood", "energy", "valence", "danceability", "acousticness"):
            v = getattr(self, key)
            if v is not None:
                prefs[key] = v
        return prefs


class RecommendationCard(BaseModel):
    song_id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    valence: float
    danceability: float
    acousticness: float
    tempo_bpm: float
    score: float
    deterministic_reasons: str
    explanation: str


class AgentTurn(BaseModel):
    query: str
    parsed_intent: IntentSchema
    cards: List[RecommendationCard]
    used_baseline_fallback: bool = False
    notes: List[str] = Field(default_factory=list)
