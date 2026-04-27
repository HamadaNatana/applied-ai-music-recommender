"""Optional Spotify Web API enrichment.

If SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET are present in the env we use
the still-working /search endpoint to look up album art and 30s preview
URLs for tracks already in the catalog. We do NOT use the audio-features,
recommendations, or related-artists endpoints — Spotify deprecated those for
new apps in November 2024.

If creds are not set, every public function returns None — the rest of the
app proceeds without enrichment. No error, no log spam.
"""

from __future__ import annotations

import os
import time
from typing import Dict, Optional

import requests

CLIENT_ID = os.environ.get("SPOTIFY_CLIENT_ID")
CLIENT_SECRET = os.environ.get("SPOTIFY_CLIENT_SECRET")

_token_cache: Dict[str, object] = {"value": None, "expires_at": 0.0}
_search_cache: Dict[str, Optional[Dict]] = {}


def has_credentials() -> bool:
    return bool(CLIENT_ID) and bool(CLIENT_SECRET)


def _get_token() -> Optional[str]:
    if not has_credentials():
        return None
    now = time.time()
    cached = _token_cache.get("value")
    expires = float(_token_cache.get("expires_at", 0.0))
    if cached and now < expires:
        return str(cached)
    try:
        resp = requests.post(
            "https://accounts.spotify.com/api/token",
            data={"grant_type": "client_credentials"},
            auth=(str(CLIENT_ID), str(CLIENT_SECRET)),
            timeout=5,
        )
        resp.raise_for_status()
        data = resp.json()
        _token_cache["value"] = data["access_token"]
        _token_cache["expires_at"] = now + float(data.get("expires_in", 3600)) - 60.0
        return str(_token_cache["value"])
    except Exception:
        return None


def search_track(title: str, artist: str) -> Optional[Dict]:
    """Look up `title` by `artist`. Returns dict with album_art_url,
    preview_url, spotify_url, or None on failure / no creds / no match."""
    cache_key = f"{title}::{artist}".lower()
    if cache_key in _search_cache:
        return _search_cache[cache_key]

    token = _get_token()
    if not token:
        _search_cache[cache_key] = None
        return None

    try:
        resp = requests.get(
            "https://api.spotify.com/v1/search",
            params={
                "q": f'track:"{title}" artist:"{artist}"',
                "type": "track",
                "limit": 1,
            },
            headers={"Authorization": f"Bearer {token}"},
            timeout=5,
        )
        resp.raise_for_status()
        items = resp.json().get("tracks", {}).get("items", [])
        if not items:
            _search_cache[cache_key] = None
            return None
        track = items[0]
        images = track.get("album", {}).get("images", []) or []
        result = {
            "spotify_id": track.get("id"),
            "album_art_url": images[0]["url"] if images else None,
            "preview_url": track.get("preview_url"),
            "spotify_url": track.get("external_urls", {}).get("spotify"),
        }
        _search_cache[cache_key] = result
        return result
    except Exception:
        _search_cache[cache_key] = None
        return None
