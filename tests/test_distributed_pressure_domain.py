"""P0身份/平均约束分派及旧HS/节点摘要逐字节兼容回归。"""
from copy import deepcopy
import unittest
from bevel_gear.pressure_background import pressure_domain_fingerprint, pressure_domain_background
from bevel_gear.distributed_pressure_domain import (distributed_body_identity,
    distributed_response_fingerprint, distributed_average_kkt_evidence)


def p0_result():
    body = dict(displacements_m=[[0., 0., 0.], [1e-6, 0., 0.]], nodal_forces_n=[[0., 0., 0.], [1., 0., 0.]],
        strain_energy_j=5e-7, external_work_j=1e-6, maximum_free_residual_n=0., relative_free_residual=0.)
    field = dict(tooth_offset=0, pressure_pa=[[100.]], forces_n=[[1.]], valid_mask=[[True]],
        contact_mask=[[True]], cell_area_m2=.01, elastic_deflection_m=[[1e-6]])
    return dict(converged=True, contact_representation='distributed_P0_surface_cells',
        pressure_filter_applied=False, halfspace_compliance_used=False,
        requested_tooth_offsets=[0, 1], selected_tooth_offsets=[0], fields=[field], cells=[],
        body_fields=dict(pinion=deepcopy(body), wheel=deepcopy(body)), body_response_checks={},
        geometry=dict(driving_angle_rad=.1, pinion_torque_nm=64., pinion_to_wheel_ratio=.5,
            reference_follower_angle_rad=.2, patches=[dict(tooth_offset=0), dict(tooth_offset=1)]),
        linearization_step_theta_rad=-.001, loaded_follower_angle_rad=.199, elastic_te_rad=-.01,
        quadrature_order=4, active_cell_keys=[(0, 0, 0)], geometry_linearization_checked=True,
        geometry_check=dict(actual_loaded_minimum_gauss_gap_m=-2e-7),
        equilibrium=dict(forces_n=[1.], remaining_gaps_m=[0.], certificate=dict(passed=True,
            force_acceptance_n=1e-8, gap_acceptance_m=1e-10, complementarity_acceptance_nm=1e-9)))


def legacy_results():
    nodal = dict(converged=True, driving_angle_rad=.1, pinion_torque_nm=64., mesh_reference_frames={'reference': 1},
        elastic_te_rad=.001, pair_force_shares={0: 1., 1: 0.}, nodal_contacts=[dict(tooth_offset=0,
        first_pressure_node_index=3, first_point_world_m=[1., 2., 3.], second_point_world_m=[1., 2., 3.1],
        normal_world=[0., 0., 1.], signed_gap_m=.1, remaining_gap_m=0., force_n=12., pressure_pa=120., area_m2=.1, active=True)], contacts=[])
    field = dict(pressure_pa=[[1.]], forces_n=[[2.]], valid_mask=[[True]], contact_mask=[[True]], cell_area_m2=.01)
    patch = dict(tooth_offset=0, frames={'reference': 1}, grid={'size': 1}, diagnostics={}, gaps_m=[[.1]],
        valid_mask=[[True]], first_points_world_m=[[[1., 2., 3.]]], first_corner_points_world_m=[[]], corner_valid_mask=[[True]])
    halfspace = dict(converged=True, selected_tooth_offsets=[0], fields=[field], theta_rad=.001,
        geometry=dict(driving_angle_rad=.2, pinion_torque_nm=64., build_parameters={'tooth_offsets': [0, 1]},
            patches=[patch, dict(patch, tooth_offset=1)]))
    return nodal, halfspace


class DistributedPressureDomainTests(unittest.TestCase):
    def test_legacy_nonempty_and_empty_fingerprints_unchanged(self):
        nodal, halfspace = legacy_results()
        expected = [('full_gear_fe', nodal, 0, '547ad890c0f9b9f709aec52a5346d7c6eb0ab9684916101978841f5e3b83a62b'),
            ('full_gear_fe', nodal, 1, 'a2791cd5e32a71131068dbf5efd0974b98d2a233ba53b7ccada3b4049752d5b9'),
            ('local_halfspace', halfspace, 0, '3cc6fb47b535a1cdb832ca9629d622a6010e021f6af89d1442ea5382299cc882'),
            ('local_halfspace', halfspace, 1, '72803bbc41a9af5146eb58d446e77a56087a9f7a0b8fe4039da9bcce15a02b53')]
        for method, result, k, digest in expected:
            self.assertEqual(pressure_domain_fingerprint(result, method=method, tooth_offset=k), digest)

    def test_p0_does_not_need_nodal_protocol(self):
        result = p0_result()
        self.assertNotIn('nodal_contacts', result)
        self.assertEqual(len(pressure_domain_fingerprint(result, method='full_gear_fe')), 64)

    def test_empty_requested_field_binds_global_actual_response(self):
        result = p0_result()
        first = pressure_domain_fingerprint(result, method='full_gear_fe', tooth_offset=1)
        result['body_fields']['wheel']['displacements_m'][1][0] *= 2
        self.assertNotEqual(first, pressure_domain_fingerprint(result, method='full_gear_fe', tooth_offset=1))
        with self.assertRaises(ValueError):
            pressure_domain_fingerprint(result, method='full_gear_fe', tooth_offset=999)

    def test_final_pose_cannot_double_add_total_te(self):
        result = p0_result()
        result['loaded_follower_angle_rad'] += result['elastic_te_rad']
        with self.assertRaises(ValueError):
            distributed_response_fingerprint(result)

    def test_external_response_identity_is_exact_and_tamper_detected(self):
        result = p0_result()
        first = distributed_response_fingerprint(result)
        result['distributed_body_identity'] = distributed_body_identity(result)
        del result['body_fields']
        self.assertEqual(first, distributed_response_fingerprint(result))
        result = p0_result()
        result['distributed_body_identity'] = distributed_body_identity(result)
        result['body_fields']['pinion']['nodal_forces_n'][1][0] = 2.
        with self.assertRaises(ValueError):
            distributed_response_fingerprint(result)

    def test_weak_average_kkt_never_certifies_gauss_pointwise(self):
        result = p0_result()
        result['minimum_gauss_remaining_gap_m'] = -3e-7
        info = distributed_average_kkt_evidence(result)
        self.assertTrue(info['passed'])
        self.assertFalse(info['pointwise_nonpenetration_certified'])
        self.assertEqual(info['actual_loaded_gauss_minimum_gap_m'], -2e-7)

    def test_uncertified_p0_background_is_unknown_with_discrete_scope(self):
        info = pressure_domain_background(p0_result(), [[[0., 0.], [1., 0.], [0., 1.]]], method='full_gear_fe')
        self.assertTrue(info['unknown_domain_present'])
        self.assertFalse(info['full_flank_zero_certified'])
        self.assertFalse(info['pointwise_nonpenetration_certified'])


if __name__ == '__main__':
    unittest.main()
