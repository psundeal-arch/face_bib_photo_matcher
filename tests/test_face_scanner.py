import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import face_scanner as fs


class TestFaceScanner(unittest.TestCase):
    def test_assign_face_ids_in_place_groups_by_distance(self) -> None:
        images = [
            {
                "status": "ok",
                "faces": [
                    {"_embedding": [0.0, 0.0], "face_hash": "h1"},
                    {"_embedding": [0.01, 0.01], "face_hash": "h2"},
                ],
            },
            {
                "status": "ok",
                "faces": [
                    {"_embedding": [2.0, 2.0], "face_hash": "h3"},
                ],
            },
        ]

        fs.assign_face_ids_in_place(images, face_threshold=0.2)

        self.assertEqual(images[0]["faces"][0]["face_id"], "face_0001")
        self.assertEqual(images[0]["faces"][1]["face_id"], "face_0001")
        self.assertEqual(images[1]["faces"][0]["face_id"], "face_0002")
        self.assertNotIn("_embedding", images[0]["faces"][0])
        self.assertIn("embedding", images[0]["faces"][0])

    def test_build_face_index_counts_samples_and_hashes(self) -> None:
        images = [
            {
                "status": "ok",
                "faces": [
                    {"face_id": "face_0001", "face_hash": "a"},
                    {"face_id": "face_0001", "face_hash": "a"},
                    {"face_id": "face_0001", "face_hash": "b"},
                ],
            },
            {
                "status": "ok",
                "faces": [{"face_id": "face_0002", "face_hash": "x"}],
            },
        ]
        idx = fs.build_face_index(images)
        self.assertEqual(idx["face_0001"]["samples"], 3)
        self.assertEqual(idx["face_0001"]["face_hashes"]["a"], 2)
        self.assertEqual(idx["face_0001"]["face_hashes"]["b"], 1)
        self.assertEqual(idx["face_0002"]["samples"], 1)

    def test_process_image_returns_error_when_cv2_unavailable(self) -> None:
        dummy_face_app = types.SimpleNamespace(get=lambda _: [])
        with patch.object(fs, "cv2", None):
            result = fs.process_image(Path("/tmp/nonexistent.jpg"), dummy_face_app, enable_bib_ocr=False)
        self.assertEqual(result["status"], "error")
        self.assertIn("OpenCV unavailable", result["error"])

    def test_process_image_happy_path_with_mocked_dependencies(self) -> None:
        class FakeDetected:
            def __init__(self, bbox, emb, score=0.99):
                self.bbox = bbox
                self.normed_embedding = np.array(emb, dtype=np.float32)
                self.det_score = score

        class FakeFaceApp:
            def get(self, _img):
                return [FakeDetected([10, 20, 40, 60], [0.1, 0.2, 0.3])]

        fake_img = np.zeros((80, 120, 3), dtype=np.uint8)
        fake_cv2 = types.SimpleNamespace(imread=lambda _: fake_img)
        with patch.object(fs, "cv2", fake_cv2), patch.object(fs, "_ocr_bib_candidates", return_value={}):
            result = fs.process_image(Path("/tmp/fake.jpg"), FakeFaceApp(), enable_bib_ocr=True)

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["face_count"], 1)
        self.assertEqual(len(result["faces"]), 1)
        self.assertIn("face_hash", result["faces"][0])
        self.assertIn("_embedding", result["faces"][0])
        self.assertIn("bib_numbers_near_face", result["faces"][0])
        self.assertEqual(result["bib_numbers"], [])


if __name__ == "__main__":
    unittest.main()
