"""Mathematical checks for the public fusion component."""

from itertools import permutations, product
import math
import unittest

from fusion import Opinion, fuse_supports


class FusionTests(unittest.TestCase):
    def test_hand_calculated_mixed_endpoints(self):
        result = fuse_supports((1.0, 0.0, 1.0))
        self.assertEqual(result.trust, 0.0)
        self.assertAlmostEqual(result.uncertainty, 0.05)
        self.assertAlmostEqual(result.conflict, 0.6016666666666667)
        self.assertAlmostEqual(result.fused_opinion.belief, 0.6333333333333333)
        self.assertAlmostEqual(result.fused_opinion.disbelief, 0.31666666666666665)

    def test_equal_ambiguous_supports(self):
        result = fuse_supports((0.5, 0.5, 0.5))
        self.assertEqual(result.trust, 0.5)
        self.assertAlmostEqual(result.uncertainty, 0.35)
        self.assertAlmostEqual(result.conflict, 0.21125)
        self.assertAlmostEqual(result.fused_opinion.belief, 0.325)
        self.assertAlmostEqual(result.fused_opinion.disbelief, 0.325)
        self.assertAlmostEqual(result.fused_opinion.projected_probability, 0.5)

    def test_equal_supports_preserve_trust_and_opinion(self):
        for value in (0.0, 0.1, 0.5, 0.9, 1.0):
            with self.subTest(value=value):
                result = fuse_supports((value,) * 3)
                self.assertAlmostEqual(result.trust, value)
                self.assertAlmostEqual(
                    result.fused_opinion.uncertainty,
                    result.source_opinions[0].uncertainty,
                )
                if value in (0.0, 1.0):
                    self.assertEqual(result.conflict, 0.0)

    def test_source_permutation_does_not_change_aggregate(self):
        expected = fuse_supports((0.95, 0.9, 0.1))
        for supports in permutations((0.95, 0.9, 0.1)):
            actual = fuse_supports(supports)
            for field in ("trust", "uncertainty", "conflict"):
                self.assertAlmostEqual(getattr(actual, field), getattr(expected, field))

    def test_conflict_is_diagnostic_not_a_trust_penalty(self):
        supports = (0.2, 0.5, 1.0)
        low_uncertainty = fuse_supports(supports, 0.1, 0.1)
        high_uncertainty = fuse_supports(supports, 0.8, 0.8)
        self.assertAlmostEqual(low_uncertainty.trust, 0.375)
        self.assertEqual(low_uncertainty.trust, high_uncertainty.trust)
        self.assertGreater(low_uncertainty.conflict, high_uncertainty.conflict)

    def test_probability_mass_and_output_bounds(self):
        for supports in product((0.0, 0.1, 0.5, 0.9, 1.0), repeat=3):
            result = fuse_supports(supports)
            self.assertGreaterEqual(result.trust, min(supports))
            self.assertLessEqual(result.trust, max(supports))
            for opinion in (*result.source_opinions, result.fused_opinion):
                self.assertAlmostEqual(
                    opinion.belief + opinion.disbelief + opinion.uncertainty, 1.0,
                )
            for value in (result.trust, result.uncertainty, result.conflict):
                self.assertTrue(math.isfinite(value))
                self.assertGreaterEqual(value, 0.0)
                self.assertLessEqual(value, 1.0)
            self.assertEqual(result.uncertainty, result.fused_opinion.uncertainty)

    def test_positive_subnormal_support_does_not_overflow(self):
        tiny = math.ulp(0.0)
        self.assertEqual(fuse_supports((tiny, tiny, tiny)).trust, tiny)
        result = fuse_supports((tiny, 1.0, 1.0))
        self.assertGreater(result.trust, 0.0)
        self.assertLessEqual(result.trust, 3.0 * tiny)

    def test_tiny_uncertainty_bounds_do_not_overflow(self):
        tiny = math.ulp(0.0)
        result = fuse_supports((0.0, 0.5, 1.0), tiny, tiny)
        self.assertEqual(result.uncertainty, tiny)
        self.assertAlmostEqual(result.fused_opinion.belief, 0.5)

    def test_complete_uncertainty(self):
        result = fuse_supports((0.1, 0.5, 0.9), 1.0, 1.0)
        self.assertEqual(result.uncertainty, 1.0)
        self.assertEqual(result.fused_opinion.belief, 0.0)
        self.assertEqual(result.fused_opinion.disbelief, 0.0)
        self.assertEqual(result.conflict, 0.0)

    def test_invalid_supports(self):
        for values in ((), (0.5,), (0.5,) * 2, (0.5,) * 4, None):
            with self.subTest(values=values), self.assertRaises(ValueError):
                fuse_supports(values)
        for value in (-0.01, 1.01, math.nan, math.inf, -math.inf, "0.5", True, 10**1000):
            with self.subTest(value=value), self.assertRaises(ValueError):
                fuse_supports((0.5, value, 0.5))

    def test_invalid_uncertainty_bounds(self):
        for lower, upper in (
            (0.0, 0.35), (-0.1, 0.35), (0.5, 0.4), (0.05, 1.01),
            (math.nan, 0.35), (0.05, math.nan), (math.inf, math.inf),
            ("0.05", 0.35), (True, 1.0),
        ):
            with self.subTest(bounds=(lower, upper)), self.assertRaises(ValueError):
                fuse_supports((0.1, 0.5, 0.9), lower, upper)

    def test_invalid_opinion(self):
        for values in ((0.5, 0.5, 0.5), (math.nan, 0.5, 0.5), (-0.1, 0.6, 0.5)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                Opinion(*values)


if __name__ == "__main__":
    unittest.main()
