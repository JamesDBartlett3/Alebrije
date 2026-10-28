import unittest

from alebrije import places


class TestMunicipalityChain(unittest.TestCase):
    def test_full_chain(self):
        chain = places.municipality_chain({
            "city": "City of Golden", "county": "Jefferson County",
            "state": "Colorado", "country": "USA", "road": "Washington Ave"})
        self.assertEqual(chain, ["City of Golden", "Jefferson County",
                                 "Colorado", "USA"])

    def test_no_duplicates(self):
        chain = places.municipality_chain({"city": "Denver", "country": "Denver"})
        self.assertEqual(chain, ["Denver"])

    def test_village_fallback(self):
        chain = places.municipality_chain({"village": "Xochimilco", "country": "México"})
        self.assertEqual(chain, ["Xochimilco", "México"])


class TestFormatPlace(unittest.TestCase):
    def test_poi_plus_chain(self):
        name = places.format_place({"name": "Washington Park", "distance_m": 40},
                                   {"city": "Denver", "state": "Colorado",
                                    "country": "USA"})
        self.assertEqual(name, "Washington Park — Denver, Colorado, USA")

    def test_chain_only_when_no_poi(self):
        name = places.format_place(None, {"city": "City of Golden",
                                          "county": "Jefferson County",
                                          "state": "Colorado", "country": "USA"})
        self.assertEqual(name, "City of Golden, Jefferson County, Colorado, USA")

    def test_poi_only_when_no_address(self):
        name = places.format_place({"name": "Chapultepec Castle"}, {})
        self.assertEqual(name, "Chapultepec Castle")

    def test_nothing_known(self):
        self.assertIsNone(places.format_place(None, {}))


class TestNearestPoi(unittest.TestCase):
    def _fake_overpass(self, elements):
        def fake(query, timeout=30.0):
            return {"elements": elements}
        return fake

    def test_picks_nearest(self):
        orig = places._post_overpass
        places._post_overpass = self._fake_overpass([
            {"type": "node", "lat": 0.0, "lon": 0.0, "tags": {"name": "Far Fountain",
                                                             "amenity": "fountain"}},
            {"type": "way", "lat": None, "center": {"lat": 0.0001, "lon": 0.0001},
             "tags": {"name": "Near Park", "leisure": "park"}},
        ])
        try:
            poi = places.nearest_poi(0.0, 0.0)
        finally:
            places._post_overpass = orig
        self.assertEqual(poi["name"], "Near Park")
        self.assertLess(poi["distance_m"], 50)

    def test_none_when_offline(self):
        orig = places._post_overpass
        places._post_overpass = lambda query, timeout=30.0: None
        try:
            self.assertIsNone(places.nearest_poi(0.0, 0.0))
        finally:
            places._post_overpass = orig

    def test_skips_unnamed(self):
        orig = places._post_overpass
        places._post_overpass = self._fake_overpass([
            {"type": "node", "lat": 0.0, "lon": 0.0, "tags": {"amenity": "bench"}}])
        try:
            self.assertIsNone(places.nearest_poi(0.0, 0.0))
        finally:
            places._post_overpass = orig


class TestGeocode(unittest.TestCase):
    def test_offline_returns_none(self):
        orig = places._get_json
        places._get_json = lambda url, timeout=20.0: None
        try:
            self.assertIsNone(places.geocode(39.7, -104.9))
        finally:
            places._get_json = orig

    def test_full_flow(self):
        replies = {
            "nominatim": {"address": {"city": "Denver", "state": "Colorado",
                                      "country": "USA"}},
            "overpass": {"elements": [{"type": "node", "lat": 39.7, "lon": -104.9,
                                       "tags": {"name": "Civic Center Park",
                                                "leisure": "park"}}]},
        }

        def fake_post(query, timeout=30.0):
            return replies["overpass"]

        def fake_get(url, timeout=20.0):
            return replies["nominatim"]

        orig_get, orig_post = places._get_json, places._post_overpass
        places._get_json, places._post_overpass = fake_get, fake_post
        try:
            name = places.geocode(39.7, -104.9)
        finally:
            places._get_json, places._post_overpass = orig_get, orig_post
        self.assertEqual(name, "Civic Center Park — Denver, Colorado, USA")


class TestBackfill(unittest.TestCase):
    def test_backfill_fills_and_respects_limit(self):
        import json as jsonlib
        from alebrije.tests import sample_db
        conn, tmp = sample_db()
        for i in range(3):
            conn.execute(
                "INSERT INTO media (path, filename, media_type, size, mtime, fingerprint,"
                " gps_lat, gps_lon) VALUES (?,?,?,?,?,?,?,?)",
                (f"/p{i}.jpg", f"p{i}.jpg", "image", 1, 1.0, f"fp{i}", 40.0, -105.0))
        conn.commit()
        calls = {"n": 0}

        def fake_geocode(lat, lon, timeout=20.0):
            calls["n"] += 1
            return f"Place {calls['n']}"

        orig = places.geocode
        places.geocode = fake_geocode
        orig_sleep = places.time.sleep
        places.time.sleep = lambda *_: None
        try:
            n = places.backfill(conn, {}, limit=2, log=lambda *_: None)
        finally:
            places.geocode = orig
            places.time.sleep = orig_sleep
        self.assertEqual(n, 2)
        self.assertEqual(calls["n"], 2)
        named = conn.execute("SELECT place FROM media ORDER BY id").fetchall()
        self.assertEqual(named[0]["place"], "Place 1")
        self.assertIsNone(named[2]["place"])
        conn.close()


if __name__ == "__main__":
    unittest.main()
