import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import build_person_index as bpi


class TestDbscanCosine(unittest.TestCase):
    def test_two_tight_clusters_plus_noise(self) -> None:
        # Two well-separated directions (each repeated), plus one lone outlier.
        a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        outlier = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        X = np.vstack([a, a, a, b, b, b, outlier])

        labels = bpi.dbscan_cosine(X, cosine_min=0.9, min_samples=3)

        # First three share a label, next three share another, they differ.
        self.assertEqual(labels[0], labels[1])
        self.assertEqual(labels[1], labels[2])
        self.assertEqual(labels[3], labels[4])
        self.assertNotEqual(labels[0], labels[3])
        # The lone point is noise.
        self.assertEqual(labels[6], -1)

    def test_empty_input(self) -> None:
        labels = bpi.dbscan_cosine(np.empty((0, 3), dtype=np.float32), 0.5, 3)
        self.assertEqual(labels.size, 0)


class TestBuildPersons(unittest.TestCase):
    def _face(self, emb, det, url, name="a.jpg", bibs=None):
        return {
            "embedding": np.asarray(emb, dtype=np.float32),
            "det_score": float(det),
            "source_url": url,
            "image_name": name,
            "image_bibs": bibs or [],
        }

    def test_representative_is_highest_det_and_photos_deduped(self) -> None:
        faces = [
            self._face([1, 0], 0.7, "https://x/p1", "p1.jpg"),
            self._face([1, 0], 0.9, "https://x/p2", "p2.jpg"),
            self._face([1, 0], 0.8, "https://x/p1", "p1.jpg"),  # same photo as first
        ]
        labels = np.array([0, 0, 0])
        persons = bpi.build_persons(faces, labels)

        self.assertEqual(len(persons), 1)
        person = persons[0]
        self.assertEqual(person["person_id"], "person_0001")
        self.assertEqual(person["face_count"], 3)
        self.assertEqual(person["photo_count"], 2)  # p1 deduped
        # Representative comes from the highest det_score face (0.9 -> p2).
        self.assertEqual(person["representative"]["image_name"], "p2.jpg")
        # Unknown host: preview/download URLs pass through verbatim.
        self.assertEqual(person["representative"]["preview_url"], "https://x/p2")


    def test_persons_sorted_by_photo_count(self) -> None:
        faces = [
            self._face([1, 0], 0.9, "https://x/a1"),
            self._face([0, 1], 0.9, "https://x/b1"),
            self._face([0, 1], 0.9, "https://x/b2"),
        ]
        labels = np.array([0, 1, 1])
        persons = bpi.build_persons(faces, labels)
        # Cluster 1 has 2 photos, cluster 0 has 1 -> the 2-photo person ranks first.
        self.assertEqual(persons[0]["photo_count"], 2)
        self.assertEqual(persons[0]["person_id"], "person_0001")
        self.assertEqual(persons[1]["photo_count"], 1)

    def test_noise_faces_excluded(self) -> None:
        faces = [self._face([1, 0], 0.9, "https://x/a1")]
        labels = np.array([-1])
        self.assertEqual(bpi.build_persons(faces, labels), [])


class TestPhotoUrls(unittest.TestCase):
    def test_google_urls_get_suffixes(self) -> None:
        import photo_urls as pu

        g = "https://lh3.googleusercontent.com/pw/ABC123"
        self.assertEqual(pu.photo_preview_url(g), g + "=w800-h560-no")
        self.assertEqual(pu.photo_download_url(g), g + "=d")
        self.assertTrue(pu.supports_face_crop(g))

    def test_runsignup_urls_swap_size_prefix(self) -> None:
        import photo_urls as pu

        r = "https://rsu-photos-v2-v2prod.s3.amazonaws.com/large_v3/race_1_2_abc.jpg"
        self.assertIn("/thumbs_v3/", pu.photo_preview_url(r))
        self.assertEqual(pu.photo_download_url(r), r)  # large_v3 is the download size
        self.assertFalse(pu.supports_face_crop(r))


class TestVoteBib(unittest.TestCase):
    def _f(self, url, bibs):
        return {"source_url": url, "near_bibs": [{"number": n, "confidence": c} for n, c in bibs]}

    def test_support_across_photos_beats_single_confident_read(self) -> None:
        faces = [
            self._f("https://x/1", [("682", 0.6)]),
            self._f("https://x/2", [("682", 0.7)]),
            self._f("https://x/3", [("999", 0.99)]),  # one very confident misread
        ]
        best, cands = bpi._vote_bib(faces)
        self.assertEqual(best["number"], "682")
        self.assertEqual(best["support"], 2)
        self.assertEqual([c["number"] for c in cands], ["682", "999"])

    def test_same_photo_counts_once_for_support(self) -> None:
        faces = [self._f("https://x/1", [("123", 0.9), ("123", 0.9)])]
        best, _ = bpi._vote_bib(faces)
        self.assertEqual(best["support"], 1)

    def test_no_reads_returns_none(self) -> None:
        self.assertEqual(bpi._vote_bib([self._f("https://x/1", [])]), (None, []))


if __name__ == "__main__":
    unittest.main()
