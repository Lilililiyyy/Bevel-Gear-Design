"""验证穿越、内部边边最近点、真实孔洞与BVH完整候选；可选依赖缺失时跳过。"""
import importlib.util
import unittest

from bevel_gear.triangle_surface_distance import (
    TriangleSurfaceDistance, engineering_clearance_margin, triangle_triangle_distance_m,
)

HAS_GEOMETRY = importlib.util.find_spec('numpy') is not None and importlib.util.find_spec('vtk') is not None


class TestEngineeringMargin(unittest.TestCase):
    def test_declared_mesh_settings_do_not_become_continuous_proof(self):
        result = engineering_clearance_margin(8e-6, 3e-6,
            source_allowance_m=1e-6, target_allowance_m=2e-6, numerical_tolerance_m=1e-10)
        self.assertAlmostEqual(result['remaining_margin_m'], 1.9999e-6, places=15)
        self.assertTrue(result['separated_under_declared_allowances'])
        self.assertFalse(result['strict_continuous_cad_proof'])


@unittest.skipUnless(HAS_GEOMETRY, '需要可选NumPy/VTK几何依赖')
class TestTriangleSurfaceDistance(unittest.TestCase):
    def test_crossing_without_vertex_on_other_face(self):
        # 第二片穿越第一片内部，六个顶点都不位于对面三角形上。
        a = [[-1., -1., 0.], [1., -1., 0.], [0., 1., 0.]]
        b = [[.1, -.2, -1.], [.1, -.2, 1.], [.1, .2, 1.]]
        self.assertEqual(triangle_triangle_distance_m(a, b), 0.)

    def test_minimum_at_two_edge_interiors(self):
        # 顶点到对面都较远，最近点处于两条有限边内部；仅顶点距离会失败。
        a = [[-1., 0., 0.], [1., 0., 0.], [1., .01, 0.]]
        b = [[0., -1., .1], [0., 1., .1], [.01, 1., .2]]
        self.assertAlmostEqual(triangle_triangle_distance_m(a, b), .1, places=13)

    def test_double_precision_at_small_clearance(self):
        z = .012345678912
        a = [[0., 0., z], [1., 0., z], [0., 1., z]]
        b = [[.1, .1, z+1e-7], [.2, .1, z+1e-7], [.1, .2, z+1e-7]]
        self.assertAlmostEqual(triangle_triangle_distance_m(a, b), 1e-7, delta=1e-15)

    def test_hole_not_filled_by_a_convex_hull(self):
        import numpy as np
        outside = np.array([[-2., -2., 0.], [2., -2., 0.], [2., 2., 0.], [-2., 2., 0.]])
        inside = outside*.5
        target = []
        for i in range(4):
            j = (i+1) % 4
            target.extend([[outside[i], outside[j], inside[j]], [outside[i], inside[j], inside[i]]])
        source = [[0., 0., 0.], [.1, 0., 0.], [0., .1, 0.]]
        result = TriangleSurfaceDistance(target, verify_bvh=True).query(source)
        self.assertAlmostEqual(result['distance_m'], .9, places=13)
        self.assertTrue(result['complete_bvh_candidates'])
        self.assertTrue(result['all_aabb_candidates_independently_checked'])

    def test_bvh_matches_all_pairs_and_overlapping_cell_bounds(self):
        import numpy as np
        rng = np.random.default_rng(713)
        target = rng.uniform(-.01, .01, (35, 3, 3))
        target[:, :, 2] *= .1
        engine = TriangleSurfaceDistance(target, verify_bvh=True)
        for _ in range(8):
            source = rng.uniform(-.004, .004, (3, 3))+[.014, 0., .02]
            expected = min(triangle_triangle_distance_m(source, triangle) for triangle in target)
            self.assertAlmostEqual(engine.query(source)['distance_m'], expected, delta=2e-13)

    def test_actual_micrometre_FE_pair_does_not_become_false_intersection(self):
        # 实际有限FE片：未归一化VTK快速相交分支曾误报0，OCP有限面为6.542137µm。
        a = [[.07151336376136581, -.007232814605190349, .027863954181627352],
            [.0715175227590969, -.007232837726874996, .02786175699400927],
            [.07152983437379715, -.007228947638422295, .027862080016920348]]
        b = [[.07144218076468703, -.0072678553718128975, .02785450501111429],
            [.07151171975215573, -.007217559293210607, .02790583449963846],
            [.07153504375305125, -.007259790428350561, .0278182414500256]]
        self.assertAlmostEqual(triangle_triangle_distance_m(a, b), 6.542137356541445e-6, delta=3e-15)

    def test_separation_crossing_coplanar_and_grazing_at_many_scales(self):
        import numpy as np
        a = np.asarray([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
        fixtures = [
            (np.asarray([[.1, .1, .03], [.3, .1, .03], [.1, .3, .03]]), .03),
            (np.asarray([[.2, .2, -1.], [.2, .2, 1.], [.7, .2, 1.]]), 0.),
            (np.asarray([[.1, .1, 0.], [.3, .1, 0.], [.1, .3, 0.]]), 0.),
            (np.asarray([[1., 0., 0.], [1.2, 0., 0.], [1., .2, 0.]]), 0.),
            (np.asarray([[1.02, 0., 0.], [1.2, 0., 0.], [1.02, .2, 0.]]), .02)]
        for scale in np.logspace(-8, 2, 11):
            shift = np.asarray([7., -3., 11.])*scale
            for b, expected in fixtures:
                with self.subTest(scale=scale, expected=expected):
                    answer = triangle_triangle_distance_m(a*scale+shift, b*scale+shift)
                    self.assertAlmostEqual(answer/scale, expected, delta=2e-12)
                    reverse = triangle_triangle_distance_m(b*scale+shift, a*scale+shift)
                    self.assertAlmostEqual(answer/scale, reverse/scale, delta=2e-12)

    def test_tiny_edge_through_large_triangle_interior_is_not_missed(self):
        import numpy as np
        a = np.asarray([[-1., -1., 0.], [1., -1., 0.], [0., 1., 0.]])
        for ratio in (1e-4, 1e-2, 1.):
            b = np.asarray([[.1, -.2, -1.], [.1, -.2, 1.], [.1, .2, 1.]])*ratio
            for scale in (1e-8, 1e-4, 1., 100.):
                self.assertEqual(triangle_triangle_distance_m(a*scale, b*scale), 0.)

    def test_degenerate_target_not_silently_dropped(self):
        with self.assertRaises(ValueError):
            TriangleSurfaceDistance([[[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]]])


if __name__ == '__main__':
    unittest.main()
