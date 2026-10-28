import os
import time
import unittest

from alebrije import scanner

from . import AlebrijeCase, exif_jpeg_bytes, png_bytes, sample_db


class TestExif(unittest.TestCase):
    def test_datetime_original(self):
        jpeg = exif_jpeg_bytes(dt="2021:11:02 14:30:00")
        self.assertEqual(scanner._exif_datetime_original(jpeg), "2021-11-02T14:30:00")

    def test_gps(self):
        jpeg = exif_jpeg_bytes(gps=(39.7392, -104.9903))
        self.assertEqual(scanner._exif_gps(jpeg), (39.7392, -104.9903))

    def test_gps_southern_western(self):
        jpeg = exif_jpeg_bytes(gps=(-33.86, -151.2))
        self.assertEqual(scanner._exif_gps(jpeg), (-33.86, -151.2))

    def test_both(self):
        jpeg = exif_jpeg_bytes(dt="2019:05:05 09:00:00", gps=(-15.0, 47.0))
        self.assertEqual(scanner._exif_datetime_original(jpeg), "2019-05-05T09:00:00")
        self.assertEqual(scanner._exif_gps(jpeg), (-15.0, 47.0))

    def test_plain_jpeg_no_exif(self):
        self.assertIsNone(scanner._exif_datetime_original(png_bytes()))
        self.assertIsNone(scanner._exif_gps(png_bytes()))

    def test_truncated_exif_does_not_crash(self):
        jpeg = exif_jpeg_bytes(dt="2021:11:02 14:30:00")[:40]
        self.assertIsNone(scanner._exif_datetime_original(jpeg))


class TestScanner(AlebrijeCase):
    def test_media_type_detection(self):
        self.assertEqual(scanner.media_type_of("a.JPG"), "image")
        self.assertEqual(scanner.media_type_of("b.MOV"), "video")
        self.assertEqual(scanner.media_type_of("c.mp3"), "audio")
        self.assertIsNone(scanner.media_type_of("d.pdf"))

    def test_scan_adds_and_tracks_changes(self):
        conn, tmp = sample_db()
        archive = os.path.join(self.home, "archive")
        p1 = scanner.os.path.join(archive, "party.jpg")
        p2 = scanner.os.path.join(archive, "song.mp3")
        for p, data in ((p1, png_bytes()), (p2, b"\xff\xfb" + b"ID3" + b"\x00" * 64)):
            os.makedirs(os.path.dirname(p), exist_ok=True)
            open(p, "wb").write(data)
        res = scanner.scan_archive(archive, conn)
        self.assertEqual(res["added"], 2)
        res2 = scanner.scan_archive(archive, conn)
        self.assertEqual(res2["unchanged"], 2)
        open(p1, "wb").write(png_bytes(w=5, h=5))
        res3 = scanner.scan_archive(archive, conn)
        self.assertEqual(res3["updated"], 1)
        conn.close()

    def test_scan_removes_gone_files(self):
        conn, tmp = sample_db()
        archive = os.path.join(self.home, "archive")
        p1 = os.path.join(archive, "party.jpg")
        os.makedirs(archive)
        open(p1, "wb").write(png_bytes())
        scanner.scan_archive(archive, conn)
        os.remove(p1)
        res = scanner.scan_archive(archive, conn)
        self.assertEqual(res["removed"], 1)

    def test_scan_captures_exif_date_and_gps(self):
        conn, tmp = sample_db()
        archive = os.path.join(self.home, "archive")
        os.makedirs(archive)
        open(os.path.join(archive, "geo.jpg"), "wb").write(
            exif_jpeg_bytes(dt="2021:11:02 14:30:00", gps=(39.7392, -104.9903)))
        scanner.scan_archive(archive, conn)
        row = conn.execute("SELECT * FROM media").fetchone()
        self.assertEqual(row["created_at"], "2021-11-02T14:30:00")
        self.assertAlmostEqual(row["gps_lat"], 39.7392)
        self.assertAlmostEqual(row["gps_lon"], -104.9903)
        conn.close()

    def test_hidden_files_skipped(self):
        conn, tmp = sample_db()
        archive = os.path.join(self.home, "archive")
        os.makedirs(archive)
        open(os.path.join(archive, ".hidden.jpg"), "wb").write(png_bytes())
        res = scanner.scan_archive(archive, conn)
        self.assertEqual(res["added"], 0)
        conn.close()


if __name__ == "__main__":
    unittest.main()
