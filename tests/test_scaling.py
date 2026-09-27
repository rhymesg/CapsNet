import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from MinMaxScaler import MinMaxScaler


class ErrorScalingChecks(unittest.TestCase):
    def test_rmse_uses_range_without_speed_offset(self):
        scaler = MinMaxScaler(np.array([10., 50.]), (0, 1))
        self.assertAlmostEqual(scaler.scale_error_inverse(0.25), 10.)

    def test_zero_error_stays_zero_with_symmetric_range(self):
        scaler = MinMaxScaler(np.array([10., 50.]), (-1, 1))
        self.assertEqual(scaler.scale_error_inverse(0.), 0.)
        self.assertAlmostEqual(scaler.scale_error_inverse(0.5), 10.)


if __name__ == '__main__':
    unittest.main()
