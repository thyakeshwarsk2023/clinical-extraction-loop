"""
tests/test_dual_blind_reread.py
=============================================================================
Unit tests for Stage 4 Independent Dual Blind Re-Read for K=2 Dual-Field Faults.

Tests:
  - Dual-agreement commits: Both crops independently yield blind re-reads
    agreeing with the candidate targets -> both_agreed is True.
  - Single-field agreement only: One crop agrees, but second crop is unreadable
    (blank/noisy) -> both_agreed is False, safely abstains.
  - Disagreement between crops: One crop agrees, but second crop reads a different
    value than candidate target -> both_agreed is False, safely abstains.
  - Candidate mismatch: Both crops readable but neither matches candidate pair -> abstains.
"""

import unittest
import numpy as np
import cv2
from src.stage4_enhance_extract import verify_dual_blind_reread


def _render_digit_crop(text: str, width: int = 120, height: int = 40) -> np.ndarray:
    """Renders clean high-contrast text crop for deterministic OCR verification."""
    img = np.full((height, width, 3), 255, dtype=np.uint8)
    cv2.putText(
        img,
        text,
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (0, 0, 0),
        2,
        cv2.LINE_AA
    )
    return img


class TestDualBlindReRead(unittest.TestCase):

    def setUp(self):
        self.crop_tc = _render_digit_crop("200.0")
        self.crop_ldl = _render_digit_crop("120.0")
        self.crop_deviant = _render_digit_crop("75.0")
        self.crop_blank = np.full((40, 120, 3), 255, dtype=np.uint8)

    def test_dual_agreement_commits(self):
        """
        When both crops independently yield blind reads agreeing with target1 and target2,
        verify_dual_blind_reread returns both_agreed = True.
        """
        both_agreed, details = verify_dual_blind_reread(
            self.crop_tc,
            self.crop_ldl,
            target1=200.0,
            target2=120.0,
            conf1=0.75,
            conf2=0.75
        )
        self.assertTrue(both_agreed)
        self.assertTrue(details["both_agreed"])
        self.assertTrue(details["field1_agreed"])
        self.assertTrue(details["field2_agreed"])
        self.assertAlmostEqual(details["field1_blind"], 200.0, places=1)
        self.assertAlmostEqual(details["field2_blind"], 120.0, places=1)
        self.assertGreater(details["field1_post_conf"], 0.75)
        self.assertGreater(details["field2_post_conf"], 0.75)

    def test_single_field_agreement_only_abstains(self):
        """
        Crop 1 agrees with target1 (200.0), but Crop 2 is unreadable/blank.
        Even though 1 crop agreed, it must NOT commit; both_agreed must be False.
        """
        both_agreed, details = verify_dual_blind_reread(
            self.crop_tc,
            self.crop_blank,
            target1=200.0,
            target2=120.0,
            conf1=0.75,
            conf2=0.75
        )
        self.assertFalse(both_agreed)
        self.assertFalse(details["both_agreed"])
        self.assertTrue(details["field1_agreed"])
        self.assertFalse(details["field2_agreed"])
        self.assertIsNone(details["field2_blind"])

    def test_disagreement_between_two_crops_abstains(self):
        """
        Crop 1 agrees with target1 (200.0), but Crop 2 reads 75.0 while target2 is 120.0.
        Disagreement with candidate target must cause both_agreed to be False.
        """
        both_agreed, details = verify_dual_blind_reread(
            self.crop_tc,
            self.crop_deviant,  # reads 75.0
            target1=200.0,
            target2=120.0,      # expected 120.0 -> mismatch
            conf1=0.75,
            conf2=0.75
        )
        self.assertFalse(both_agreed)
        self.assertFalse(details["both_agreed"])
        self.assertTrue(details["field1_agreed"])
        self.assertFalse(details["field2_agreed"])
        self.assertAlmostEqual(details["field2_blind"], 75.0, places=1)

    def test_both_crops_disagree_abstains(self):
        """Both crops read values disagreeing with candidate targets."""
        both_agreed, details = verify_dual_blind_reread(
            self.crop_deviant,  # reads 75.0 vs 200.0
            self.crop_blank,    # blank vs 120.0
            target1=200.0,
            target2=120.0
        )
        self.assertFalse(both_agreed)
        self.assertFalse(details["field1_agreed"])
        self.assertFalse(details["field2_agreed"])


if __name__ == "__main__":
    unittest.main()
