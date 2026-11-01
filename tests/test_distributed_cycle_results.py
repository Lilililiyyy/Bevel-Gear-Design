"""独立核验从动角直接差分定义、原pA守恒和拒绝相位断点。"""
import copy
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest

from bevel_gear.distributed_fe_results import summarize_distributed_fe_cycle,load_distributed_fe_cycle


def accepted_state():
    angle=.1;ratio=17/47;nominal=-angle*ratio
    return dict(schema='bevel_distributed_fe_state_v1',phase_index=3,driving_angle_rad=angle,accepted=True,
        result=dict(converged=True,geometry_linearization_checked=True,
            contact_representation='distributed_P0_surface_cells',halfspace_compliance_used=False,
            equilibrium=dict(certificate=dict(passed=True)),
            geometry=dict(driving_angle_rad=angle,pinion_to_wheel_ratio=ratio,
                reference_tca=dict(follower_increment_angle_rad=nominal+2e-5)),
            unloaded_reference_follower_angle_rad=nominal+2e-5+1e-10,
            loaded_follower_angle_rad=nominal+2e-5+1e-10-4e-4,elastic_te_rad=-4e-4,
            incremental_stiffness_nm_per_rad=3e5,pinion_torque_relative_residual=1e-5,
            minimum_gauss_remaining_gap_m=-1e-7,total_force_plane_n=6.,contact_area_plane_m2=3e-6,
            maximum_pressure_pa=3e6,selected_tooth_offsets=[-1,0],fields=[
                dict(tooth_offset=-1,cell_area_m2=1e-6,pressure_pa=[[1e6,2e6]],forces_n=[[1.,2.]],contact_mask=[[True,True]]),
                dict(tooth_offset=0,cell_area_m2=1e-6,pressure_pa=[[3e6,0.]],forces_n=[[3.,0.]],contact_mask=[[True,False]])]))


class CycleResultsTests(unittest.TestCase):
    def test_follower_angle_and_P0_integral(self):
        state=accepted_state();row=summarize_distributed_fe_cycle([state])['rows'][0]
        direct=state['result']['loaded_follower_angle_rad']+state['driving_angle_rad']*17/47
        self.assertAlmostEqual(row['total_te_rad'],direct,places=15)
        self.assertAlmostEqual(row['unloaded_te_rad'],2e-5,places=14)
        self.assertAlmostEqual(row['reference_phase_correction_rad'],1e-10,places=14)
        self.assertEqual(row['active_P0_area_plane_m2'],3e-6)
        self.assertEqual([p['force_plane_fraction'] for p in row['pair_force_plane_shares']],[.5,.5])
        altered=copy.deepcopy(state);altered['result']['fields'][0]['pressure_pa'][0][0]*=2
        with self.assertRaises(ArithmeticError):summarize_distributed_fe_cycle([altered])

    def test_rejected_phase_is_gap_and_duplicate_is_error(self):
        bad=dict(schema='bevel_distributed_fe_state_v1',phase_index=2,driving_angle_rad=0.,accepted=False,
            result=dict(converged=False,status='not_converged'))
        data=summarize_distributed_fe_cycle([accepted_state(),bad])
        self.assertFalse(data['selected_states_completed']);self.assertEqual(data['accepted_state_count'],1)
        self.assertNotIn('elastic_te_rad',data['rows'][0])
        self.assertFalse(data['phase_resolution_converged'])
        with self.assertRaises(ValueError):summarize_distributed_fe_cycle([bad,bad])
        wrong=accepted_state();wrong['result']['loaded_follower_angle_rad']+=1e-4
        with self.assertRaises(ArithmeticError):summarize_distributed_fe_cycle([wrong])

    def test_source_state_hash_and_input_binding(self):
        with tempfile.TemporaryDirectory() as folder:
            directory=Path(folder);state=accepted_state()
            state.update(input_fingerprint='fixed_input',source_identity_unchanged=True)
            raw=json.dumps(state).encode('utf-8');(directory/'state.json').write_bytes(raw)
            row=dict(phase_index=3,driving_angle_rad=.1,accepted=True,state_file='state.json',
                state_sha256=hashlib.sha256(raw).hexdigest())
            summary=dict(schema='bevel_distributed_fe_workflow_v1',input_fingerprint='fixed_input',states=[row])
            path=directory/'summary.json';path.write_text(json.dumps(summary),encoding='utf-8')
            self.assertEqual(load_distributed_fe_cycle(path)['accepted_state_count'],1)
            (directory/'state.json').write_bytes(raw+b' ')
            with self.assertRaises(ValueError):load_distributed_fe_cycle(path)
            (directory/'state.json').write_bytes(raw)
            summary['input_fingerprint']='changed_input';path.write_text(json.dumps(summary),encoding='utf-8')
            with self.assertRaises(ValueError):load_distributed_fe_cycle(path)

    def test_native_total_angle_is_not_last_increment(self):
        state=accepted_state();r=state['result'];total=r['loaded_follower_angle_rad']+.1*17/47
        r.update(geometry_source='FE_boundary',total_loaded_te_rad=total,
            reference_follower_angle_rad=r['loaded_follower_angle_rad']-2e-11,linearization_step_theta_rad=2e-11,
            geometry_check=dict(loaded_FE_phi_geometry_check=dict(minimum_remaining_gap_m=-1e-8)))
        for key in ('elastic_te_rad','unloaded_reference_follower_angle_rad','minimum_gauss_remaining_gap_m'):r.pop(key)
        r['geometry'].pop('reference_tca')
        data=summarize_distributed_fe_cycle([state]);row=data['rows'][0]
        self.assertEqual(data['geometry_source'],'FE_boundary')
        self.assertAlmostEqual(row['total_te_rad'],total)
        self.assertEqual(row['linearization_step_theta_rad'],2e-11)
        self.assertNotIn('elastic_te_rad',row);self.assertNotIn('unloaded_te_rad',row)
        self.assertEqual(row['minimum_piece_vertex_gap_m'],-1e-8)
        wrong=copy.deepcopy(state);wrong['result']['total_loaded_te_rad']+=1e-5
        with self.assertRaises(ArithmeticError):summarize_distributed_fe_cycle([wrong])
        wrong=copy.deepcopy(state);wrong['result']['reference_follower_angle_rad']+=1e-5
        with self.assertRaises(ArithmeticError):summarize_distributed_fe_cycle([wrong])


if __name__=='__main__':unittest.main()
