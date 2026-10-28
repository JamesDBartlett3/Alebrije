"""Turn EXIF GPS coordinates into human place names.

Two open-data services, no keys, and the flow degrades gracefully offline:

1. **Closest point of interest** — the Overpass API finds named POIs
   (parks, museums, landmarks, restaurants, trails, anything tagged
   tourism/leisure/historic/amenity) within 200 m, and we take the
   *nearest* one, computed from real geometry rather than trusting
   whichever feature Nominatim happens to snap to.
2. **Municipality chain** — always appended, POI or not:
   "City Park — Denver, City and County of Denver, Colorado, USA", or with
   no POI in range simply "City of Golden, Jefferson County, Colorado, USA".

Results are cached in the media table; rate limits are respected (1 req/s).
"""

from __future__ import annotations

import json
import math
import time
import urllib.error
import urllib.parse
import urllib.request

OVERPASS = "https://overpass-api.de/api/interpreter"
NOMINATIM = "https://nominatim.openstreetmap.org/reverse"
POI_RADIUS_M = 200
UA = "alebrije/0.1 (local family archive; respectful, 1 req/s)"

# Overpass tags we treat as "a place worth naming", landmark-ish first.
POI_FILTERS = ["historic", "tourism", "leisure", "amenity"]

# Nominatim address keys for the municipality chain, most specific first.
MUNICIPALITY_KEYS = ["city", "town", "village", "hamlet", "municipality",
                     "borough", "suburb", "county", "state", "region"]


def _haversine_m(lat1, lon1, lat2, lon2) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 6371000.0 * 2 * math.asin(math.sqrt(a))


def _get_json(url: str, timeout: float = 20.0) -> dict | None:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, TimeoutError, json.JSONDecodeError):
        return None


def nearest_poi(lat: float, lon: float, radius_m: int = POI_RADIUS_M) -> dict | None:
    """Nearest named POI to (lat, lon) via Overpass. None if nothing in range."""
    around = f"(around:{radius_m},{lat},{lon})"
    tests = "|".join(f"node[{t}][name]{around};way[{t}][name]{around};" for t in POI_FILTERS)
    query = f"[out:json][timeout:20];({tests});out center 20;"
    data = _post_overpass(query)
    if data is None:
        return None
    best = None
    best_d = float("inf")
    for el in data.get("elements", []):
        name = (el.get("tags") or {}).get("name")
        if not name:
            continue
        plat = el.get("lat") or (el.get("center") or {}).get("lat")
        plon = el.get("lon") or (el.get("center") or {}).get("lon")
        if plat is None or plon is None:
            continue
        d = _haversine_m(lat, lon, plat, plon)
        if d < best_d:
            best, best_d = {"name": name, "distance_m": round(d)}, d
    return best


def _post_overpass(query: str, timeout: float = 30.0) -> dict | None:
    body = urllib.parse.urlencode({"data": query}).encode()
    req = urllib.request.Request(OVERPASS, body, {
        "Content-Type": "application/x-www-form-urlencoded", "User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, TimeoutError, json.JSONDecodeError):
        return None


def municipality_chain(address: dict) -> list[str]:
    chain = []
    for k in MUNICIPALITY_KEYS:
        v = address.get(k)
        if v and v not in chain:
            chain.append(v)
    country = address.get("country")
    if country and country not in chain:
        chain.append(country)
    return chain


def format_place(poi: dict | None, address: dict) -> str | None:
    """'POI — City, County, State, Country' or just the chain."""
    chain = municipality_chain(address or {})
    if poi and chain:
        return f"{poi['name']} — {', '.join(chain)}"
    if poi:
        return poi["name"]
    if chain:
        return ", ".join(chain)
    return None


def geocode(lat: float, lon: float, timeout: float = 20.0) -> str | None:
    """Closest POI + always the municipality chain. None if offline."""
    url = (NOMINATIM + "?" + urllib.parse.urlencode(
        {"lat": lat, "lon": lon, "format": "jsonv2", "zoom": "16",
         "addressdetails": "1"}))
    data = _get_json(url, timeout)
    if data is None:
        return None
    poi = nearest_poi(lat, lon)
    return format_place(poi, data.get("address") or {})


def backfill(conn, cfg: dict, limit: int = 40, log=print) -> int:
    """Reverse-geocode media rows that have GPS but no place name yet."""
    rows = conn.execute(
        "SELECT id, gps_lat, gps_lon FROM media "
        "WHERE gps_lat IS NOT NULL AND (place IS NULL OR place='') "
        "ORDER BY id LIMIT ?", (limit,)).fetchall()
    if not rows:
        return 0
    done = 0
    for row in rows:
        name = geocode(row["gps_lat"], row["gps_lon"])
        if name:
            conn.execute("UPDATE media SET place=? WHERE id=?", (name, row["id"]))
            conn.commit()
            done += 1
            log(f"📍 {name}")
        else:
            log("• place lookup unavailable (offline?) — will retry later")
        time.sleep(1.1)  # Nominatim/Overpass usage policy: ~1 req/s
    return done
