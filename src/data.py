"""Catalog data layer.

At first run we try to download a real Spotify tracks dataset from
HuggingFace Hub, normalize genres, derive a `mood` column from audio
features, sample to a working subset, and cache the result as parquet.

If the HF download fails (no network, hub unreachable, etc.) we fall back
to data/songs_sample.csv — a small curated catalog committed to the repo so
the project is fully runnable offline and tests don't need the network.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
PARQUET_PATH = DATA_DIR / "songs.parquet"
SAMPLE_CSV_PATH = DATA_DIR / "songs_sample.csv"
LEGACY_CSV_PATH = DATA_DIR / "songs.csv"

HF_DATASET_ID = os.environ.get(
    "SOUNDFIT_HF_DATASET", "maharshipandya/spotify-tracks-dataset"
)
TARGET_CATALOG_SIZE = int(os.environ.get("SOUNDFIT_CATALOG_SIZE", "5000"))

# Spotify genre strings are very granular. Map each into a top-level family
# the intent parser can reason about. Anything not in this map falls back to
# the genre's first word and ultimately to "other" if still no match.
GENRE_FAMILIES: Dict[str, str] = {
    # pop
    "pop": "pop", "indie pop": "pop", "synth-pop": "pop", "k-pop": "pop",
    "j-pop": "pop", "power-pop": "pop", "dance-pop": "pop", "indie": "pop",
    # rock
    "rock": "rock", "alt-rock": "rock", "alternative": "rock", "punk": "rock",
    "punk-rock": "rock", "hard-rock": "rock", "psych-rock": "rock",
    "grunge": "rock", "garage": "rock", "rock-n-roll": "rock",
    # metal
    "metal": "metal", "heavy-metal": "metal", "death-metal": "metal",
    "black-metal": "metal", "metalcore": "metal",
    # hip-hop
    "hip-hop": "hip-hop", "rap": "hip-hop", "trap": "hip-hop",
    "j-rap": "hip-hop", "k-rap": "hip-hop",
    # electronic
    "electronic": "electronic", "edm": "electronic", "house": "electronic",
    "deep-house": "electronic", "techno": "electronic", "trance": "electronic",
    "drum-and-bass": "electronic", "dubstep": "electronic", "synthwave": "electronic",
    "electro": "electronic", "ambient": "electronic", "chill": "electronic",
    "lofi": "electronic", "lo-fi": "electronic", "idm": "electronic",
    # jazz
    "jazz": "jazz", "blues": "jazz", "bebop": "jazz", "swing": "jazz",
    # classical
    "classical": "classical", "opera": "classical", "piano": "classical",
    "orchestral": "classical",
    # country
    "country": "country", "bluegrass": "country", "honky-tonk": "country",
    # r&b / soul
    "r-n-b": "r&b", "r&b": "r&b", "soul": "r&b", "funk": "r&b",
    "neo-soul": "r&b", "motown": "r&b",
    # latin
    "latin": "latin", "reggaeton": "latin", "salsa": "latin", "bossa-nova": "latin",
    "samba": "latin", "tango": "latin",
    # folk / acoustic
    "folk": "folk", "acoustic": "folk", "singer-songwriter": "folk",
    "country-rock": "folk", "americana": "folk",
    # reggae
    "reggae": "reggae", "ska": "reggae", "dub": "reggae",
}

ALL_GENRE_FAMILIES = sorted(set(GENRE_FAMILIES.values()))
ALL_MOODS = ["happy", "chill", "intense", "moody", "relaxed", "focused"]


def normalize_genre(raw: Optional[str]) -> str:
    """Map Spotify's raw genre strings into a clean top-level family."""
    if not raw:
        return "other"
    key = raw.strip().lower().replace("_", "-")
    if key in GENRE_FAMILIES:
        return GENRE_FAMILIES[key]
    # try first word of multi-word genre (e.g. "japanese-trap" -> "trap")
    for piece in key.split("-"):
        if piece in GENRE_FAMILIES:
            return GENRE_FAMILIES[piece]
    # try direct family match
    if key in ALL_GENRE_FAMILIES:
        return key
    return "other"


def derive_mood(row: pd.Series) -> str:
    """Map (valence, energy, acousticness, instrumentalness) -> mood label.

    This is a transparent, hand-tuned mapping documented in model_card.md.
    Real Spotify data has no `mood` field — the educational scorer needs one,
    so we synthesize it deterministically from audio features.
    """
    valence = float(row.get("valence", 0.5) or 0.5)
    energy = float(row.get("energy", 0.5) or 0.5)
    acousticness = float(row.get("acousticness", 0.5) or 0.5)
    instrumentalness = float(row.get("instrumentalness", 0.0) or 0.0)

    if instrumentalness > 0.5 and energy < 0.5: return "focused"
    if valence < 0.4 and energy >= 0.6: return "intense"
    if valence < 0.4 and energy < 0.4: return "moody"
    if valence < 0.5 and acousticness > 0.6: return "relaxed"
    if valence >= 0.6 and energy >= 0.6: return "happy"
    if valence >= 0.5 and energy < 0.55: return "chill"
    if energy >= 0.7: return "intense"
    return "chill"


def _popularity_weights(group: pd.DataFrame) -> Optional[pd.Series]:
    """Bias sampling toward popular tracks so famous artists land in the
    sample rather than getting cut by uniform random. Quadratic so a
    popularity-80 track is ~50x more likely than a popularity-10 track."""
    if "popularity" not in group.columns:
        return None
    base = group["popularity"].fillna(0).clip(0, 100).astype(float) + 10.0
    return base ** 2


def _stratified_sample(
    df: pd.DataFrame, target_size: int, per_genre_min: int = 100
) -> pd.DataFrame:
    """Sample so every genre family has at least `per_genre_min` tracks
    (or all of them if fewer exist) up to `target_size` total rows.
    Sampling is weighted by Spotify popularity so household-name artists
    (Kendrick, Beyoncé, Drake, etc.) make it into the working set."""
    if len(df) <= target_size:
        return df.copy()
    parts: List[pd.DataFrame] = []
    by_genre = df.groupby("genre", group_keys=False)
    for _, group in by_genre:
        n = min(len(group), per_genre_min)
        weights = _popularity_weights(group)
        parts.append(group.sample(n=n, weights=weights, random_state=42))
    base = pd.concat(parts)
    remaining = target_size - len(base)
    if remaining > 0:
        leftover = df.drop(base.index, errors="ignore")
        n_extra = min(remaining, len(leftover))
        if n_extra > 0:
            extras = leftover.sample(
                n=n_extra,
                weights=_popularity_weights(leftover),
                random_state=42,
            )
            base = pd.concat([base, extras])
    return base.sample(frac=1.0, random_state=42).reset_index(drop=True)


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Bring HF dataset columns into the schema the scorer expects."""
    df = df.copy()
    rename = {
        "track_id": "spotify_id",
        "track_name": "title",
        "artists": "artist",
        "track_genre": "genre_raw",
        "tempo": "tempo_bpm",
    }
    for src_col, dst_col in rename.items():
        if src_col in df.columns and dst_col not in df.columns:
            df[dst_col] = df[src_col]
    needed = ["title", "artist", "genre_raw", "energy", "valence",
              "danceability", "acousticness", "tempo_bpm"]
    for col in needed:
        if col not in df.columns:
            df[col] = None
    df["title"] = df["title"].fillna("Unknown").astype(str)
    df["artist"] = df["artist"].fillna("Unknown").astype(str).str.split(";").str[0]
    df = df.dropna(subset=["energy", "valence", "danceability", "acousticness"])
    df["genre"] = df["genre_raw"].apply(normalize_genre)
    df = df[df["genre"] != "other"].reset_index(drop=True)
    df["mood"] = df.apply(derive_mood, axis=1)
    df["id"] = range(1, len(df) + 1)
    keep_cols = ["id", "title", "artist", "genre", "genre_raw", "mood",
                 "energy", "tempo_bpm", "valence", "danceability", "acousticness"]
    if "popularity" in df.columns:
        df["popularity"] = df["popularity"].fillna(0).astype(int)
        keep_cols.append("popularity")
    if "spotify_id" in df.columns:
        keep_cols.append("spotify_id")
    return df[keep_cols]


def _load_from_huggingface() -> Optional[pd.DataFrame]:
    try:
        from datasets import load_dataset
    except Exception:
        return None
    try:
        ds = load_dataset(HF_DATASET_ID, split="train")
        df = ds.to_pandas()
        df = _normalize_columns(df)
        df = _stratified_sample(df, TARGET_CATALOG_SIZE)
        df["id"] = range(1, len(df) + 1)
        return df
    except Exception as exc:
        print(f"[data] HF download failed ({exc}); using local sample.")
        return None


def _load_sample_csv() -> pd.DataFrame:
    if SAMPLE_CSV_PATH.exists():
        df = pd.read_csv(SAMPLE_CSV_PATH)
        if "genre_raw" not in df.columns:
            df["genre_raw"] = df["genre"]
        return df
    # Last-resort: use the original tiny 10-song catalog
    df = pd.read_csv(LEGACY_CSV_PATH)
    df["genre_raw"] = df["genre"]
    return df


def load_catalog_df(force_refresh: bool = False) -> pd.DataFrame:
    """Return the working catalog as a pandas DataFrame.

    Order of preference:
      1. cached parquet at data/songs.parquet (fast)
      2. fresh HuggingFace download (cached to parquet)
      3. data/songs_sample.csv (offline fallback)
      4. data/songs.csv (legacy, last resort)
    """
    if PARQUET_PATH.exists() and not force_refresh:
        return pd.read_parquet(PARQUET_PATH)

    df = _load_from_huggingface()
    if df is not None and len(df) > 0:
        try:
            DATA_DIR.mkdir(exist_ok=True)
            df.to_parquet(PARQUET_PATH)
        except Exception as exc:
            print(f"[data] failed to cache parquet: {exc}")
        return df

    return _load_sample_csv()


def load_catalog_records(force_refresh: bool = False) -> List[Dict]:
    """Return the working catalog as a list of dicts (the shape the scorer
    in src/recommender.py expects)."""
    df = load_catalog_df(force_refresh=force_refresh)
    return df.to_dict("records")


def get_allowed_genres(songs: List[Dict]) -> List[str]:
    return sorted({str(s["genre"]) for s in songs if s.get("genre")})


def get_allowed_moods() -> List[str]:
    return list(ALL_MOODS)
