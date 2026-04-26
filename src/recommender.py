from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
import csv

@dataclass
class Song:
    """
    Represents a song and its attributes.
    Required by tests/test_recommender.py
    """
    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: float
    valence: float
    danceability: float
    acousticness: float

@dataclass
class UserProfile:
    """
    Represents a user's taste preferences.
    Required by tests/test_recommender.py
    """
    favorite_genre: str
    favorite_mood: str
    target_energy: float
    likes_acoustic: bool


def _profile_to_prefs(user: UserProfile) -> Dict:
    return {
        "genre": user.favorite_genre,
        "mood": user.favorite_mood,
        "energy": user.target_energy,
        "acousticness": 0.8 if user.likes_acoustic else 0.2,
    }


class Recommender:
    """
    OOP implementation of the recommendation logic.
    Delegates to the functional core (score_song / recommend_songs) so the
    scoring math has a single source of truth.
    """
    def __init__(self, songs: List[Song]):
        self.songs = songs

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        prefs = _profile_to_prefs(user)
        song_dicts = [asdict(s) for s in self.songs]
        ranked = recommend_songs(prefs, song_dicts, k=k)
        by_id = {s.id: s for s in self.songs}
        return [by_id[r[0]["id"]] for r in ranked]

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        prefs = _profile_to_prefs(user)
        _, reasons = score_song(prefs, asdict(song))
        if not reasons:
            return f"{song.title} by {song.artist}: no matching preferences."
        return f"{song.title} by {song.artist} — " + "; ".join(reasons)

def load_songs(csv_path: str) -> List[Dict]:
    """
    Loads songs from a CSV file.
    Required by src/main.py
    """
    print(f"Loading songs from {csv_path}...")
    songs = []
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            row['id'] = int(row['id'])
            for key in ['energy', 'tempo_bpm', 'valence', 'danceability', 'acousticness']:
                row[key] = float(row[key])
            songs.append(row)
    return songs

def score_song(user_prefs: Dict, song: Dict) -> Tuple[float, List[str]]:
    """
    Scores a single song against user preferences.
    Required by recommend_songs() and src/main.py
    """
    score = 0.0
    reasons = []

    # Genre match (weight: 3)
    if 'genre' in user_prefs:
        if song['genre'] == user_prefs['genre']:
            score += 3
            reasons.append(f"Genre match: {song['genre']}")

    # Mood match (weight: 2)
    if 'mood' in user_prefs:
        if song['mood'] == user_prefs['mood']:
            score += 2
            reasons.append(f"Mood match: {song['mood']}")

    # Energy closeness (weight: 2)
    if 'energy' in user_prefs:
        energy_closeness = 1 - abs(user_prefs['energy'] - song['energy'])
        score += 2 * energy_closeness
        reasons.append(f"Energy closeness: {energy_closeness:.2f}")

    # Valence closeness (weight: 1.5)
    if 'valence' in user_prefs:
        valence_closeness = 1 - abs(user_prefs['valence'] - song['valence'])
        score += 1.5 * valence_closeness
        reasons.append(f"Valence closeness: {valence_closeness:.2f}")

    # Danceability (weight: 1)
    if 'danceability' in user_prefs:
        dance_closeness = 1 - abs(user_prefs['danceability'] - song['danceability'])
        score += 1 * dance_closeness
        reasons.append(f"Danceability closeness: {dance_closeness:.2f}")

    # Acousticness (weight: 0.5)
    if 'acousticness' in user_prefs:
        acoustic_closeness = 1 - abs(user_prefs['acousticness'] - song['acousticness'])
        score += 0.5 * acoustic_closeness
        reasons.append(f"Acousticness closeness: {acoustic_closeness:.2f}")

    return (score, reasons)

def recommend_songs(user_prefs: Dict, songs: List[Dict], k: int = 5) -> List[Tuple[Dict, float, str]]:
    """
    Functional implementation of the recommendation logic.
    Required by src/main.py
    """
    scored = [
        (song, score, "; ".join(reasons))
        for song in songs
        for score, reasons in [score_song(user_prefs, song)]
    ]
    return sorted(scored, key=lambda x: x[1], reverse=True)[:k]
