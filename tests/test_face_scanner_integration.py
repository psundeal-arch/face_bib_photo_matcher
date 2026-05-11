import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import face_scanner as fs

FIXTURE_IMAGE = ROOT / "tests" / "fixtures" / "BE7I4001.JPG"


@unittest.skipUnless(
    os.environ.get("RUN_INTEGRATION_TESTS") == "1",
    "Set RUN_INTEGRATION_TESTS=1 to run integration tests",
)
class TestFaceScannerIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not FIXTURE_IMAGE.exists():
            raise unittest.SkipTest(f"Fixture missing: {FIXTURE_IMAGE}")
        if not fs.is_cv2_available():
            raise unittest.SkipTest("OpenCV unavailable")
        if not fs.is_insightface_available():
            raise unittest.SkipTest("InsightFace unavailable")

        cls.face_app = fs.create_insightface_app(
            det_size=320,
            model_name=fs.DEFAULT_INSIGHTFACE_MODEL_NAME,
            provider_mode="cpu",
        )

    def test_process_real_fixture_image(self) -> None:
        result = fs.process_image(
            image_path=FIXTURE_IMAGE,
            face_app=self.face_app,
            enable_bib_ocr=False,
        )

        self.assertEqual(result.get("status"), "ok")
        self.assertEqual(result.get("file"), str(FIXTURE_IMAGE))
        self.assertIn("faces", result)
        self.assertIn("face_count", result)
        self.assertIsInstance(result["faces"], list)
        self.assertEqual(result["face_count"], len(result["faces"]))
        self.assertIn("bib_numbers", result)
        self.assertIsInstance(result["bib_numbers"], list)

        for face in result["faces"]:
            self.assertIn("face_hash", face)
            self.assertIn("_embedding", face)
            self.assertIn("det_score", face)

    def test_assign_and_index_from_real_fixture_image(self) -> None:
        result = fs.process_image(
            image_path=FIXTURE_IMAGE,
            face_app=self.face_app,
            enable_bib_ocr=False,
        )
        self.assertEqual(result.get("status"), "ok")

        images = [result]
        fs.assign_face_ids_in_place(images, face_threshold=1.0)
        idx = fs.build_face_index(images)

        self.assertIsInstance(idx, dict)
        for face in images[0].get("faces", []):
            self.assertIn("face_id", face)
            self.assertIn("embedding", face)
            self.assertNotIn("_embedding", face)


if __name__ == "__main__":
    unittest.main()
