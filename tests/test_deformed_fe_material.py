"""变形FE全材料闭合、内外与完整三角几何桥接回归。"""
import importlib.util
import unittest

HAS_NUMERIC = importlib.util.find_spec('numpy') is not None and importlib.util.find_spec('vtk') is not None


@unittest.skipUnless(HAS_NUMERIC, '需要可选NumPy/VTK几何依赖')
class DeformedFEMaterialTests(unittest.TestCase):
    def setUp(self):
        import numpy as np
        self.np = np
        self.nodes = np.asarray([[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.],
            [0., 0., 1.], [1., 0., 1.], [1., 1., 1.], [0., 1., 1.]])
        self.faces = np.asarray([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
            [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]])

    def test_closed_shared_indices_and_hole_rejected(self):
        from bevel_gear.deformed_fe_material import closed_boundary_evidence
        self.assertTrue(closed_boundary_evidence(self.nodes, self.faces)['passed'])
        with self.assertRaises(ValueError):
            closed_boundary_evidence(self.nodes, self.faces[:-1])
        with self.assertRaises(ValueError):
            closed_boundary_evidence(self.nodes, self.np.concatenate((self.faces, self.faces[:1])))

    def test_positive_unsigned_inside_distance_cannot_prove_outside(self):
        from bevel_gear.deformed_fe_material import ClosedTriangleMaterial
        material = ClosedTriangleMaterial(self.nodes, self.faces)
        inner = self.np.asarray([[.2, .2, .2], [.3, .2, .2], [.2, .3, .2]])
        evidence = material.triangle_evidence(inner)
        self.assertGreater(evidence['distance_m'], .19)
        self.assertFalse(evidence['outside'])
        self.assertEqual(evidence['material_state'], 'IN')
        outer = inner+self.np.asarray([0., 0., 1.])
        self.assertTrue(material.triangle_evidence(outer)['outside'])

    def test_whole_triangle_bridge_keeps_valid_upper_at_resource_limit(self):
        from bevel_gear.deformed_fe_material import triangle_to_surface_upper_bound
        from bevel_gear.triangle_surface_distance import TriangleSurfaceDistance
        target = TriangleSurfaceDistance(self.nodes[self.faces])
        source = self.np.asarray([[.25, .25, 1.01], [.75, .25, 1.01], [.25, .75, 1.01]])
        result = triangle_to_surface_upper_bound(source, target, maximum_depth=0)
        self.assertGreaterEqual(result['distance_upper_bound_m'], .01)
        if result['projected_FE_cover'] and result['projected_FE_cover']['passed']:
            self.assertLess(result['distance_upper_bound_m'], .011)
        refined = triangle_to_surface_upper_bound(source, target, accuracy_m=1e-3, maximum_depth=9)
        self.assertGreaterEqual(refined['distance_upper_bound_m'], .01)
        self.assertLess(refined['distance_upper_bound_m'], .011)

    def test_complete_carrier_candidates_include_both_sides_of_fe_edge(self):
        from bevel_gear.deformed_fe_material import complete_trace_carriers
        from bevel_gear.triangle_surface_distance import TriangleSurfaceDistance
        target = TriangleSurfaceDistance(self.nodes[self.faces])
        source = self.np.asarray([[.4, .3, 1.001], [.7, .6, 1.001], [.3, .7, 1.001]])
        result = complete_trace_carriers(source, target)
        self.assertTrue({2, 3}.issubset(result['indices']))
        self.assertTrue(result['complete_bvh_candidates'])

    def test_actual_body_translation_changes_material_without_new_solve(self):
        from bevel_gear.deformed_fe_material import ClosedTriangleMaterial
        first = ClosedTriangleMaterial(self.nodes, self.faces)
        second = ClosedTriangleMaterial(self.nodes+self.np.asarray([0., 0., 2.]), self.faces)
        self.assertEqual(first.classify([.5, .5, .5]), 'IN')
        self.assertEqual(second.classify([.5, .5, .5]), 'OUT')

    def test_material_cavity_remains_outside_no_convex_hull(self):
        from bevel_gear.deformed_fe_material import ClosedTriangleMaterial
        nodes = self.np.concatenate((self.nodes, self.nodes*.5+.25))
        faces = self.np.concatenate((self.faces, self.faces[:, ::-1]+8))
        material = ClosedTriangleMaterial(nodes, faces)
        self.assertEqual(material.classify([.5, .5, .5]), 'OUT')
        self.assertEqual(material.classify([.1, .5, .5]), 'IN')

    def test_canonical_leaf_cache_same_material_key_under_different_phases(self):
        from bevel_gear.deformed_fe_material import canonical_midpoint_leaf, _split
        root = self.np.asarray([[.1, .2, .3], [.12, .2, .3], [.1, .23, .3]])
        leaf = _split(_split(root)[3])[1]
        answers = []
        for angle in (0., .123, -.217):
            c, s = self.np.cos(angle), self.np.sin(angle)
            rotation = self.np.asarray([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
            answers.append(canonical_midpoint_leaf(leaf@rotation.T, root, rotation, 2))
        for actual in answers[1:]:
            self.assertTrue(self.np.array_equal(actual, answers[0]))
        wrong = leaf.copy(); wrong[0, 0] += 1e-6
        with self.assertRaises(ValueError):
            canonical_midpoint_leaf(wrong, root, self.np.eye(3), 2)

    def test_inverted_tetrahedron_rejected_by_actual_deformation_jacobian(self):
        from bevel_gear.deformed_fe_material import deformation_jacobian_evidence
        nodes = self.np.asarray([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
        displacement = self.np.zeros_like(nodes)
        self.assertTrue(deformation_jacobian_evidence(nodes, self.np.asarray([[0, 1, 2, 3]]), displacement)['passed'])
        displacement[3, 2] = -2.
        self.assertFalse(deformation_jacobian_evidence(nodes, self.np.asarray([[0, 1, 2, 3]]), displacement)['passed'])

    def test_exact_carrier_clip_keeps_original_P1_weights(self):
        from bevel_gear.deformed_fe_material import clip_triangle_to_box_with_weights
        triangle = self.np.asarray([[0., 0., 0.], [2., 0., 0.], [0., 2., 0.]])
        polygon = clip_triangle_to_box_with_weights(triangle, [.2, .2, -.1], [.8, .8, .1])
        self.assertEqual(len(polygon), 4)
        displacement = self.np.asarray([[0., 0., .1], [.2, 0., .2], [0., .3, .3]])
        for point, weights in polygon:
            self.assertTrue(self.np.allclose(point, weights@triangle, atol=1e-14, rtol=0))
            self.assertAlmostEqual(float(weights.sum()), 1.)
            self.assertGreaterEqual(float(weights.min()), 0.)
            self.assertTrue(self.np.allclose(point+weights@displacement, weights@(triangle+displacement)))
        self.assertEqual(clip_triangle_to_box_with_weights(triangle, [3., 3., 3.], [4., 4., 4.]), [])

    @unittest.skipUnless(importlib.util.find_spec('shapely') is not None, '需要可选Shapely/GEOS')
    def test_projected_finite_FE_cover_gives_tight_bound_without_filling_holes(self):
        from bevel_gear.deformed_fe_material import projected_surface_cover_upper_bound
        from bevel_gear.triangle_surface_distance import TriangleSurfaceDistance
        flat = self.nodes[self.faces][2:4]
        target = TriangleSurfaceDistance(flat)
        source = self.np.asarray([[.25, .25, 1.01], [.75, .25, 1.01], [.25, .75, 1.01]])
        proof = projected_surface_cover_upper_bound(source, target, 1.)
        self.assertTrue(proof['passed'])
        self.assertLess(proof['distance_upper_bound_m'], .010001)
        self.assertFalse(proof['buffer_or_convex_hull_applied'])
        incomplete = TriangleSurfaceDistance(flat[:1])
        self.assertFalse(projected_surface_cover_upper_bound(source, incomplete, 1.)['passed'])

    def test_supplied_boundary_must_equal_all_single_tetrahedron_faces(self):
        from bevel_gear.deformed_fe_material import tetrahedral_boundary_evidence
        tets = self.np.asarray([[0, 1, 2, 3], [0, 2, 1, 4]])
        boundary = self.np.asarray([[0, 1, 3], [0, 2, 3], [1, 2, 3], [0, 1, 4], [0, 2, 4], [1, 2, 4]])
        proof = tetrahedral_boundary_evidence(tets, boundary)
        self.assertTrue(proof['passed'])
        self.assertEqual(proof['internal_face_count'], 1)
        self.assertFalse(tetrahedral_boundary_evidence(tets, boundary[:-1])['passed'])
        self.assertFalse(tetrahedral_boundary_evidence(tets, self.np.concatenate((boundary, [[0, 1, 2]])))['passed'])


if __name__ == '__main__':
    unittest.main()
