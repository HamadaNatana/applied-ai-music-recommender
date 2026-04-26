"""
Command line runner for the Music Recommender Simulation.

This file helps you quickly run and test your recommender.

You will implement the functions in recommender.py:
- load_songs
- score_song
- recommend_songs
"""

from src.recommender import load_songs, recommend_songs


def main() -> None:
    songs = load_songs("data/songs.csv") 
    print(f"Loaded {len(songs)} songs.")

    # Starter example profile
    user_prefs = {"genre": "pop", "mood": "happy", "energy": 0.8}
    user_prefs2 = {"genre": "lofi", "mood": "chill", "energy": 0.4}
    user_prefs3 = {"genre": "rock", "mood": "angry", "energy": 0.2}
    user_prefs4 = {"genre": "jazz", "mood": "intense", "energy": 0.8}

    lst = [
        recommend_songs(user_prefs, songs, k=5),
        recommend_songs(user_prefs2, songs, k=5),
        recommend_songs(user_prefs3, songs, k=5),
        recommend_songs(user_prefs4, songs, k=5)
    ]

    for i, recommendations in enumerate(lst):
        print(f"\nTop recommendations for User {i+1}:\n")
        for rec in recommendations:
            # You decide the structure of each returned item.
            # A common pattern is: (song, score, explanation)
            song, score, explanation = rec
            print(f"{song['title']} - Score: {score:.2f}")
            print(f"Because: {explanation}")
            print()


if __name__ == "__main__":
    main()
