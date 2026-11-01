"""点间隙声明绑定的纯协议控制；不构建FE系统，不冒充新数值结果。"""
from copy import deepcopy
import unittest

from bevel_gear.native_results_geometry import validate_original_native_geometry_gate


class DeclaredPointGapTests(unittest.TestCase):
    @staticmethod
    def geometry_result(tolerance,point_penetration):
        initial=dict(passed=True,partial_cells=0,checked_vertices=3,minimum_remaining_gap_m=-point_penetration,
            maximum_mean_vs_QP_error_m=0.,maximum_original_q_mean_vs_W_mean_error_m=0.)
        loaded=dict(passed=True,partial_cells=0,checked_vertices=3,minimum_remaining_gap_m=-point_penetration,
            finite_phi_geometry_rebuilt=True,maximum_cell_mean_error_m=0.)
        row=dict(tooth_offset=0,row=0,column=0,loaded_complete=True,old_minimum_piece_vertex_gap_m=-point_penetration,
            loaded_minimum_piece_vertex_gap_m=-point_penetration,old_mean_remaining_m=0.,reference_remaining_gap_m=0.,loaded_vs_old_predicted_mean_error_m=0.)
        return dict(geometry_check=dict(completed=True,passed=True,source_NURBS_gap_reused=False,
            tolerances=dict(mean_error_m=1e-10,penetration_m=tolerance),initial_piece_vertex_check=initial,loaded_FE_phi_geometry_check=loaded,cell_checks=[row],delta_theta_rad=0.),
            fields=[dict(tooth_offset=0,valid_mask=[[True]])],linearization_step_theta_rad=0.,equilibrium=dict(theta_rad=0.),
            reference_follower_angle_rad=0.,loaded_follower_angle_rad=0.,geometry=dict(pinion_to_wheel_ratio=17/47,driving_angle_rad=0.),total_loaded_te_rad=0.)

    def test_bound_default_explicit_range_and_original_mean_failed_gate(self):
        old=self.geometry_result(2e-7,.1e-6);old_before=deepcopy(old)
        self.assertEqual(validate_original_native_geometry_gate(old)['penetration_tolerance_m'],2e-7)
        self.assertEqual(old,old_before)
        engineering=self.geometry_result(1e-6,.323e-6);before=deepcopy(engineering)
        value=validate_original_native_geometry_gate(engineering,declared_controls=dict(maximum_piece_penetration_m=1e-6))
        self.assertEqual(value['penetration_tolerance_m'],1e-6);self.assertEqual(engineering,before)
        # reported值不代替输入声明；未声明、错声明和超过API新上限均拒绝。
        for controls in (None,{},dict(maximum_piece_penetration_m=.8e-6),dict(maximum_piece_penetration_m=1.01e-6)):
            with self.subTest(controls=controls),self.assertRaises(ValueError):
                validate_original_native_geometry_gate(engineering,declared_controls=controls)
        wrong_mean=deepcopy(engineering);wrong_mean['geometry_check']['tolerances']['mean_error_m']=2e-10
        with self.assertRaises(ValueError):validate_original_native_geometry_gate(wrong_mean,declared_controls=dict(maximum_piece_penetration_m=1e-6))
        failed=deepcopy(engineering);failed['geometry_check']['passed']=False;unchanged=deepcopy(failed)
        with self.assertRaises(ValueError):validate_original_native_geometry_gate(failed,declared_controls=dict(maximum_piece_penetration_m=1e-6))
        self.assertEqual(failed,unchanged)


if __name__=='__main__':unittest.main()
