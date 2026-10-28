import math
import os
import sys
import unittest

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "util")),
)

from laser_phoenix_lattice_pq import (
    PHI_DEN,
    PHI_NUM,
    Q16_ONE,
    LaserPhoenixPolarLattice,
    PolarFeatureBFP64,
    densify_sparse_stokes_2d_phoenix_pq,
    calc_dolp_aolp_admap_pq,
    catmull_rom_pq,
    cordic_atan2_u32,
    project_stokes_cone_pq,
    signed_phase_diff_i32,
)


class StokesConeTest(unittest.TestCase):
    def test_in_cone_input_is_preserved(self):
        s0, s1, s2, p_norm = project_stokes_cone_pq(Q16_ONE, Q16_ONE // 2, Q16_ONE // 4)
        self.assertEqual((s0, s1, s2), (Q16_ONE, Q16_ONE // 2, Q16_ONE // 4))
        self.assertLessEqual(p_norm, s0)

    def test_out_of_cone_input_is_projected_onto_boundary(self):
        s1, s2 = 3 * Q16_ONE, 4 * Q16_ONE
        s0, s1p, s2p, p_norm = project_stokes_cone_pq(Q16_ONE, s1, s2)
        self.assertLessEqual(p_norm, s0)
        mag_sq = s1p * s1p + s2p * s2p
        self.assertLessEqual(mag_sq, s0 * s0 + 2 * s0)
        self.assertAlmostEqual(s1p / s2p, s1 / s2, places=3)

    def test_unpolarized_and_zero_intensity(self):
        self.assertEqual(project_stokes_cone_pq(0, 5, 5), (0, 0, 0, 0))
        s0, s1, s2, p_norm = project_stokes_cone_pq(Q16_ONE, 0, 0)
        self.assertEqual((s0, s1, s2, p_norm), (Q16_ONE, 0, 0, 0))


class DolpAolpAdmapTest(unittest.TestCase):
    def test_fully_polarized_axis_aligned(self):
        dolp, phase, (r, g, b) = calc_dolp_aolp_admap_pq(Q16_ONE, Q16_ONE, 0)
        self.assertEqual(dolp, Q16_ONE)
        self.assertLess(signed_phase_diff_i32(phase, 0), 1 << 15)
        self.assertEqual(r, Q16_ONE)
        self.assertEqual(g, 0)
        self.assertEqual(b, 0)

    def test_unpolarized_gives_neutral_blue(self):
        dolp, _phase, (r, g, b) = calc_dolp_aolp_admap_pq(Q16_ONE, 0, 0)
        self.assertEqual(dolp, 0)
        self.assertEqual((r, g, b), (0, 0, Q16_ONE))

    def test_rg_match_float_ratio(self):
        s0, s1, s2 = Q16_ONE, Q16_ONE // 3, Q16_ONE // 5
        _dolp, _phase, (r, g, b) = calc_dolp_aolp_admap_pq(s0, s1, s2)
        self.assertAlmostEqual(r / Q16_ONE, s1 / s0, delta=2e-4)
        self.assertAlmostEqual(g / Q16_ONE, s2 / s0, delta=2e-4)
        rem = math.sqrt(max(0.0, 1.0 - (s1 / s0) ** 2 - (s2 / s0) ** 2))
        self.assertAlmostEqual(b / Q16_ONE, rem, delta=2e-4)

    def test_respects_realizability_cone(self):
        _dolp, _phase, (r, g, b) = calc_dolp_aolp_admap_pq(
            Q16_ONE, 2 * Q16_ONE, 2 * Q16_ONE
        )
        for c in (r, g, b):
            self.assertGreaterEqual(c, -Q16_ONE)
            self.assertLessEqual(c, Q16_ONE)


class CordicTest(unittest.TestCase):
    def test_matches_math_atan2_across_quadrants(self):
        for k in range(-8, 9):
            angle = math.pi * k / 8.0
            x = int(round(1000 * math.cos(angle)))
            y = int(round(1000 * math.sin(angle)))
            got = cordic_atan2_u32(y, x)
            want = (2 * math.pi * ((math.atan2(y, x)) / (2 * math.pi))) % (2 * math.pi)
            err = abs(signed_phase_diff_i32(got, int(want / (2 * math.pi) * (1 << 32))))
            self.assertLess(err, 1 << 22, f"angle={angle}")

    def test_origin_is_zero(self):
        self.assertEqual(cordic_atan2_u32(0, 0), 0)

    def test_phase_diff_wraps(self):
        self.assertEqual(signed_phase_diff_i32(0, 0xFFFFFFFF), 1)
        self.assertEqual(signed_phase_diff_i32(0xFFFFFFFF, 0), -1)


class CatmullRomTest(unittest.TestCase):
    def test_endpoints(self):
        self.assertEqual(catmull_rom_pq(2, 5, 9, 11, 0, 4), 5)
        self.assertEqual(catmull_rom_pq(2, 5, 9, 11, 4, 4), 9)

    def test_midpoint_is_between_control_points(self):
        v = catmull_rom_pq(0, 0, 100, 100, 2, 4)
        self.assertGreaterEqual(v, 0)
        self.assertLessEqual(v, 100)

    def test_matches_float_catmull_rom(self):
        p0, p1, p2, p3 = -3, 7, 11, 20
        for num, den in ((1, 4), (1, 2), (3, 4)):
            t = num / den
            want = 0.5 * (
                2 * p1
                + (-p0 + p2) * t
                + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t
                + (-p0 + 3 * p1 - 3 * p2 + p3) * t * t * t
            )
            got = catmull_rom_pq(p0, p1, p2, p3, num, den)
            self.assertLess(abs(got - want), 1.0, f"t={num}/{den}")


class DensificationMseTest(unittest.TestCase):
    @staticmethod
    def _smooth_field(h, w):
        return [
            [
                int(round(Q16_ONE * 0.5 * math.sin(0.35 * r) + Q16_ONE * 0.4 * math.cos(0.27 * c)))
                for c in range(w)
            ]
            for r in range(h)
        ]

    @staticmethod
    def _mse(grid, truth):
        h, w = len(truth), len(truth[0])
        return sum((grid[r][c] - truth[r][c]) ** 2 for r in range(h) for c in range(w)) / (h * w)

    def _gain(self, mask_fn):
        truth = self._smooth_field(32, 32)
        mask = [[mask_fn(r, c) for c in range(32)] for r in range(32)]
        sparse = [[truth[r][c] if mask[r][c] else 0 for c in range(32)] for r in range(32)]
        dense = densify_sparse_stokes_2d_phoenix_pq(sparse, mask)
        return self._mse(sparse, truth) / self._mse(dense, truth)

    def test_x4_2x2_superpixel_gain_exceeds_50x(self):
        gain = self._gain(lambda r, c: r % 4 < 2 and c % 4 < 2)
        self.assertGreater(gain, 50.0, f"measured {gain:.1f}x")

    def test_x16_diagonal_anchor_gain_exceeds_20x(self):
        gain = self._gain(lambda r, c: r % 4 == 0 and c % 4 == 0)
        self.assertGreater(gain, 20.0, f"measured {gain:.1f}x")


class PhiHarmonicTest(unittest.TestCase):
    def test_phi_pair_detected(self):
        self.assertTrue(LaserPhoenixPolarLattice.is_phi_harmonic_pair(PHI_DEN, PHI_NUM))

    def test_unrelated_pair_rejected(self):
        self.assertFalse(LaserPhoenixPolarLattice.is_phi_harmonic_pair(100, 151))
        self.assertFalse(LaserPhoenixPolarLattice.is_phi_harmonic_pair(0, 5))


class Bfp64FeatureTest(unittest.TestCase):
    def test_roundtrip_within_one_lsb_of_scale(self):
        values = [Q16_ONE, -Q16_ONE // 2, Q16_ONE // 7, 0, -3]
        packed = PolarFeatureBFP64.quantize_q16(values)
        back = packed.dequantize_q16()
        step = 1 << (packed.exponent + 16)
        for original, recovered in zip(values, back):
            self.assertLessEqual(abs(original - recovered), step, f"{original}")

    def test_wire_packet_is_fixed_66_bytes(self):
        packed = PolarFeatureBFP64.quantize_q16([Q16_ONE] * 4)
        self.assertEqual(len(packed.to_bytes()), 66)
        self.assertEqual(packed.count, 4)
        self.assertAlmostEqual((64 * 4) / len(packed.to_bytes()), 3.8788, places=3)

    def test_serialization_roundtrip(self):
        packed = PolarFeatureBFP64.quantize_q16([Q16_ONE, -Q16_ONE // 3, 7])
        restored = PolarFeatureBFP64.from_bytes(packed.to_bytes())
        self.assertEqual(restored, packed)


if __name__ == "__main__":
    unittest.main()
