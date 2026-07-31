import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


class PlottingTests(unittest.TestCase):
    def test_padded_axis_limits_keep_a_minimum_display_span(self):
        from alloy_agent.plotting import padded_axis_limits

        lower, upper = padded_axis_limits([3.17, 2.91], min_span=4.0, nonnegative=True)

        self.assertEqual(lower, 0.0)
        self.assertGreaterEqual(upper - lower, 4.0)

    def test_padded_axis_limits_do_not_clip_negative_values(self):
        from alloy_agent.plotting import padded_axis_limits

        lower, upper = padded_axis_limits([-0.17, 3.17], min_span=4.0, nonnegative=True)

        self.assertLess(lower, -0.17)
        self.assertGreater(upper, 3.17)


if __name__ == "__main__":
    unittest.main()
