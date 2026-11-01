"""材料内正距离、未证桥片及近点误连接的独立回归。"""
import unittest
from bevel_gear.pressure_material_domain import audit_triangle_material_domain, exact_vertex_components


class MaterialDomainTests(unittest.TestCase):
    def setUp(self):
        self.triangles = [[[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]],
                          [[1., 0., 0.], [2., 0., 0.], [1., 1., 0.]],
                          [[2., 0., 0.], [3., 0., 0.], [3., 1., 0.]]]

    def audit(self, classifier, oracle=None, motions=None):
        if oracle is None:
            oracle = lambda i, triangle: dict(done=True, complete_target_search=True, distance_lower_bound_m=2.)
        return audit_triangle_material_domain(self.triangles, [1.]*3 if motions is None else motions,
            distance_lower_oracle=oracle, material_classifier=classifier,
            boundary_sha256='a'*64, boundary_is_closed=True)

    def test_connected_outside_requires_only_one_actual_classifier_query(self):
        probes = []
        result = self.audit(lambda point: (probes.append(point) or dict(state='outside')))
        self.assertTrue(result['passed'])
        self.assertEqual(len(probes), 1)
        self.assertEqual(result['components'][0]['triangle_indices'], [0, 1, 2])

    def test_positive_unsigned_distance_in_material_is_never_zero_far(self):
        result = self.audit(lambda point: dict(state='inside', native_state='IN'))
        self.assertFalse(result['passed'])
        self.assertEqual(result['unresolved_triangle_indices'], [0, 1, 2])

    def test_incomplete_target_search_cannot_certify(self):
        result = self.audit(lambda point: self.fail('不完整候选域不应调用材料传播'),
            oracle=lambda i, t: dict(done=True, complete_target_search=False, distance_lower_bound_m=200.))
        self.assertEqual(result['components'], [])
        self.assertEqual(result['unresolved_triangle_indices'], [0, 1, 2])

    def test_unproven_bridge_does_not_join_components(self):
        result = self.audit(lambda point: dict(state='outside'),
            oracle=lambda i, t: dict(done=i != 1, complete_target_search=True, distance_lower_bound_m=2.))
        self.assertEqual([c['triangle_indices'] for c in result['components']], [[0], [2]])
        self.assertEqual(result['unresolved_triangle_indices'], [1])

    def test_no_tolerance_snapping(self):
        self.triangles[1][0][0] += 1e-15
        self.assertEqual(exact_vertex_components(self.triangles, [0, 1]), [[0], [1]])

    def test_equal_motion_margin_stays_unresolved(self):
        result = self.audit(lambda point: self.fail('零余量不能传播材料分类'), motions=[2.]*3)
        self.assertFalse(result['passed'])
        self.assertEqual(result['unresolved_triangle_indices'], [0, 1, 2])

    def test_open_boundary_and_nonfinite_lower_rejected(self):
        with self.assertRaises(ValueError):
            audit_triangle_material_domain(self.triangles, [1.]*3,
                distance_lower_oracle=lambda i, t: {}, material_classifier=lambda p: {},
                boundary_sha256='a'*64, boundary_is_closed=False)
        with self.assertRaises(ValueError):
            self.audit(lambda point: dict(state='outside'),
                oracle=lambda i, t: dict(done=True, complete_target_search=True, distance_lower_bound_m=float('nan')))

    def test_index_and_geometry_validation(self):
        with self.assertRaises(ValueError):
            exact_vertex_components(self.triangles, [0, 0])
        with self.assertRaises(ValueError):
            exact_vertex_components([[[0., 0., 0.]]*3])


if __name__ == '__main__':
    unittest.main()
