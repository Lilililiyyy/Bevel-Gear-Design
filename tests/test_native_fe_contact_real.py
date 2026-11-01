"""真实两体TET4、生产QP和outer的数值回归；没有替换任何求解函数。"""
import unittest

import numpy as np

from examples.validate_native_contact import run_validation


class NativeContactRealNumericsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.answer=run_validation()

    def test_real_K_QP_and_independent_dense_elimination(self):
        answer=self.answer
        self.assertTrue(answer['native_contact_accepted'],answer['rejection_reasons'])
        self.assertTrue(answer['passed'])
        audit=answer['dense_reference'];C=np.asarray(audit['active_C_m_per_n'])
        self.assertEqual(C.shape,(4,4))
        self.assertGreater(float(np.min(C[np.triu_indices(4,1)])),0.)
        self.assertGreater(float(np.linalg.eigvalsh(C).min()),0.)
        self.assertEqual(audit['candidate_count'],64)
        self.assertEqual(audit['zero_pressure_count'],60)
        self.assertTrue(audit['loaded_pressure_window_has_zero_neighbor_ring'])
        self.assertFalse(audit['halfspace_or_Hertz_gap_used'])
        self.assertLess(audit['active_C_dense_vs_sparse_relative_error'],1e-10)
        self.assertLess(audit['dense_active_force_relative_error'],1e-8)
        self.assertLess(audit['final_active_stationarity_maximum_m'],1e-10)
        self.assertLess(audit['FE_vs_dense_condensed_energy_relative_error'],1e-9)
        self.assertLess(audit['reported_stiffness_relative_error'],1e-9)
        for member,row in audit['body_checks'].items():
            self.assertEqual((row['nodes'],row['tetrahedra']),(192,588))
            self.assertLess(row['dense_vs_sparse_displacement_relative_error'],1e-9)
            self.assertLess(row['original_Wf_relative_error'],1e-12)
            self.assertLess(row['free_Ku_minus_F_relative_error'],1e-9)
            self.assertLess(row['energy_work_relative_error'],1e-9)
            self.assertLess(row['nodal_vs_record_torque_relative_error'],1e-10)
            self.assertAlmostEqual(row['reconstructed_external_torque_nm'],-.003,places=10)
            self.assertEqual(row['fixed_displacement_maximum_m'],0.)
            self.assertTrue(answer['preparation_audit']['compact_binding'][member]['complete_tetrahedral_boundary_verified'])

    def test_real_piece_finite_phi_gates_and_total_TE(self):
        answer=self.answer;check=answer['geometry_check'];history=answer['geometry_outer_history']
        self.assertTrue(check['passed'])
        self.assertEqual(len(history),2)
        for name in ('initial_piece_vertex_check','loaded_FE_phi_geometry_check'):
            self.assertTrue(check[name]['passed'])
            self.assertGreater(check[name]['checked_vertices'],0)
            self.assertGreaterEqual(check[name]['minimum_remaining_gap_m'],-2e-7)
        self.assertLessEqual(check['initial_piece_vertex_check']['maximum_mean_vs_QP_error_m'],1e-10)
        self.assertLessEqual(check['loaded_FE_phi_geometry_check']['maximum_cell_mean_error_m'],1e-10)
        self.assertTrue(check['loaded_FE_phi_geometry_check']['finite_phi_geometry_rebuilt'])
        audit=answer['dense_reference'];total=sum(h['delta_theta_rad'] for h in history)
        self.assertLess(abs(audit['actual_last_delta_rad']),1e-8)
        self.assertAlmostEqual(audit['actual_total_loaded_TE_rad'],total,places=14)
        self.assertGreater(abs(audit['actual_total_loaded_TE_rad']-audit['actual_last_delta_rad']),1e-8)
        self.assertTrue(answer['input_mesh_arrays_unchanged'])

    def test_real_second_load_cold_start_and_zero_neighbor_ring(self):
        second=run_validation(torque_nm=-.006)
        self.assertTrue(second['passed'],second['rejection_reasons'])
        self.assertEqual(second['geometry_outer_history'][0]['warm_physical_key_count'],0)
        self.assertEqual(self.answer['geometry_outer_history'][0]['warm_physical_key_count'],0)
        self.assertTrue(second['dense_reference']['loaded_pressure_window_has_zero_neighbor_ring'])
        self.assertGreater(sum(second['dense_reference']['original_active_force_n']),
            sum(self.answer['dense_reference']['original_active_force_n']))
        self.assertGreater(abs(second['dense_reference']['actual_total_loaded_TE_rad']),
            abs(self.answer['dense_reference']['actual_total_loaded_TE_rad']))


if __name__=='__main__':unittest.main(verbosity=2)
