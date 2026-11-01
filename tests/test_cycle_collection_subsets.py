"""中断业务的逐状态复用：不改原完成标志、不跳过失败、不伪造周期。"""
import copy
import json
import math
from pathlib import Path
import tempfile
import unittest

from bevel_gear.cycle_collection import create_mesh_cycle_collection, load_cycle_source, load_mesh_cycle_collection
from bevel_gear.surface_workflow import _fingerprint


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding='utf-8')


def _source(root, selected, saved, completed):
    phases = [-math.pi/17+i*math.pi/34 for i in range(5)]
    base = dict(manufacturing=dict(pinion=dict(request=dict(tooth_count=17)),
        wheel=dict(request=dict(tooth_count=47))), cad_sampling={})
    analysis = dict(tooth_offsets=[-1, 0, 1])
    materials = dict(pinion_young_modulus_pa=210e9, wheel_young_modulus_pa=210e9)
    solver = dict(maximum_iterations=500)
    fingerprint = _fingerprint(dict(base_case=base, analysis=analysis,
        driver_angles_rad=phases, solver=solver, pinion_signed_torque_nm=-64., materials=materials))
    inputs = dict(base_case=base, surface_case=dict(analysis=dict(driver_angles_rad=phases)),
        effective_analysis=analysis, effective_solver=solver, materials=materials,
        effective_input_fingerprint=fingerprint, pinion_signed_torque_nm=-64.,
        maximum_pinion_torque_relative_residual=.02)
    identity = {'kernel.py': 'a'*64}
    rows = []
    for i in saved:
        result = dict(converged=True, fields=[dict(pressure_pa=[[1.]])],
            geometry_linearization_checked=True,
            geometry_outer_history=[dict(geometry_check=dict(passed=True))],
            active_contact_domain_accepted=True, pinion_torque_relative_residual=1e-6,
            geometry=dict(driving_angle_rad=phases[i], pinion_torque_nm=-64.))
        state = dict(phase_index=i, driving_angle_rad=phases[i], accepted=True,
            input_fingerprint=fingerprint, source_identity_before=identity,
            source_identity_after=identity, source_identity_unchanged=True, result=result)
        reference = f'states/phase_{i:04d}.json'
        _write(root/reference, state)
        rows.append(dict(phase_index=i, driving_angle_rad=phases[i], accepted=True, state_file=reference))
    summary = dict(schema='bevel_surface_workflow_v1', analysis_completed=completed,
        selected_phase_indices=selected, source_identity_before=identity,
        source_identity_after=identity, source_identity_unchanged=True,
        input_file='surface_inputs.json', input_fingerprint=fingerprint, states=rows)
    _write(root/'surface_inputs.json', inputs)
    _write(root/'surface_summary.json', summary)
    return root/'surface_summary.json'


class AcceptedSubsetTests(unittest.TestCase):
    def test_reuse_keeps_original_bytes_and_false_completion_and_lazy_sha(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = _source(root/'interrupted', [0, 1, 2], [0, 1], False)
            second = _source(root/'resumed', [2, 3, 4], [2, 3, 4], True)
            original = {p: p.read_bytes() for p in root.rglob('*.json')}
            with self.assertRaisesRegex(ValueError, '未完成'):
                load_cycle_source(first)
            collection = create_mesh_cycle_collection([first, second], root/'collection',
                accepted_source_subsets={0: [0, 1]})
            loaded = load_mesh_cycle_collection(collection)
            self.assertEqual(len(loaded['results']), 5)
            self.assertEqual(loaded['cycle']['interval_count'], 4)
            manifest = json.loads(collection.read_text(encoding='utf-8'))
            row = manifest['sources'][0]
            copied = collection.parent/row['summary_file']
            self.assertFalse(json.loads(copied.read_text(encoding='utf-8'))['analysis_completed'])
            self.assertEqual(row['accepted_phase_indices'], [0, 1])
            for path, raw in original.items():
                self.assertEqual(path.read_bytes(), raw)
            self.assertEqual(copied.read_bytes(), first.read_bytes())
            self.assertTrue(loaded['results'][0]['converged'])
            state = copied.parent/'states/phase_0000.json'
            state.write_bytes(state.read_bytes()+b'\n')
            with self.assertRaisesRegex(ValueError, 'SHA256'):
                loaded['results'][0]

    def test_explicit_subset_cannot_skip_gap_or_accept_missing_failed_physics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = _source(root/'interrupted', [0, 1, 2], [0, 1], False)
            for selected in ([2], [0, 0], [True], []):
                with self.subTest(selected=selected), self.assertRaises(ValueError):
                    load_cycle_source(source, accepted_phase_indices=selected)
            with self.assertRaises(ValueError):
                load_cycle_source(source, require_complete_cycle=True, accepted_phase_indices=[0, 1])
            second = _source(root/'resumed', [3, 4], [3, 4], True)
            with self.assertRaisesRegex(ValueError, '均匀|至少'):
                create_mesh_cycle_collection([source, second], root/'bad_collection',
                    accepted_source_subsets={0: [0, 1]})
            state_path = source.parent/'states/phase_0001.json'
            state = json.loads(state_path.read_text(encoding='utf-8'))
            state['result']['pinion_torque_relative_residual'] = .5
            _write(state_path, state)
            with self.assertRaisesRegex(ValueError, '接受条件'):
                load_cycle_source(source, accepted_phase_indices=[1])

    def test_original_failed_state_not_silently_filtered(self):
        with tempfile.TemporaryDirectory() as directory:
            source = _source(Path(directory)/'interrupted', [0, 1, 2], [0, 1], False)
            summary = json.loads(source.read_text(encoding='utf-8'))
            summary['states'][1]['accepted'] = False
            _write(source, summary)
            with self.assertRaisesRegex(ValueError, '未接受'):
                load_cycle_source(source, accepted_phase_indices=[0, 1])
            accepted = load_cycle_source(source, accepted_phase_indices=[0])
            self.assertEqual(accepted['identity']['accepted_phase_indices'], [0])
            self.assertFalse(accepted['identity']['source_analysis_completed'])


if __name__ == '__main__':
    unittest.main()
