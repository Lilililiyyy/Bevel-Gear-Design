"""相位入场策略的三个专项控制；不创建CAD/Gmsh/弹性矩阵。"""
from copy import deepcopy
from dataclasses import asdict,dataclass
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from bevel_gear.native_case_geometry import load_case,load_case_refinement_regions

PROJECT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('cycle_scope_native_runner',PROJECT/'examples/run_native_fe_case.py')
runner=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(runner)
SPEC2=importlib.util.spec_from_file_location('cycle_scope_regions_loader',PROJECT/'examples/run_distributed_fe_contact.py')
driver=importlib.util.module_from_spec(SPEC2);SPEC2.loader.exec_module(driver)
CASE=PROJECT/'examples/spiral_17_47_relieved_native_fe_cycle9_case.json'


class CycleBackgroundPolicyTests(unittest.TestCase):
    def test_default_requires_each_phase_local_source(self):
        loaded=load_case(CASE);loaded['case'].pop('refinement_phase_policy')
        with self.assertRaisesRegex(ValueError,'所选相位不在'):load_case_refinement_regions(loaded,driver._load_regions_file)
        loaded['selected_phase_indices']=[4]  # 同一实际alpha0；默认旧行为仍可用。
        _,_,_,scope=load_case_refinement_regions(loaded,driver._load_regions_file)
        self.assertEqual(scope['policy'],'require_local_sources');self.assertEqual(scope['phases_using_global_background'],[])

    def test_explicit_background_keeps_actual_alpha0_sources_and_region_values(self):
        loaded=load_case(CASE);before=loaded['regions_file'].read_bytes()
        regions,_,_,scope=load_case_refinement_regions(loaded,driver._load_regions_file)
        self.assertEqual(regions['source_driver_angles_rad'],[0.]);self.assertEqual(scope['source_driver_angles_rad'],[0.])
        self.assertEqual(len(scope['phases']),9);self.assertEqual(len(scope['phases_using_global_background']),8)
        self.assertEqual([r['phase_index'] for r in scope['phases'] if r['local_source_declared']],[4])
        self.assertFalse(scope['whole_cycle_local_fine_coverage_certified']);self.assertFalse(scope['spatial_error_certified'])
        self.assertTrue(scope['original_API_pre_QP_FE_coverage_required']);self.assertEqual(scope['background_track_refined_size_m'],.0001)
        self.assertEqual(regions['regions'],json.loads(before)['regions']);self.assertEqual(before,loaded['regions_file'].read_bytes())
        loaded['case']['refinement_phase_policy']='unsupported'
        with self.assertRaisesRegex(ValueError,'未知'):load_case_refinement_regions(loaded,driver._load_regions_file)

    def test_explicit_background_does_not_bypass_actual_mesh_binding(self):
        @dataclass
        class Request:tooth_count:int=17
        @dataclass
        class Blank:bore_diameter_m:float=.004
        models={m:SimpleNamespace(request=Request(),settings=Blank()) for m in ('pinion','wheel')}
        cad={m:'a'*64 for m in models};regions=dict(regions={m:[] for m in models},source_driver_angles_rad=[0.])
        background=dict(element_size_m=.001,contact_refined_size_m=.0001,refinement_core_radius_m=.0003,refinement_radius_m=.0012)
        metadata=dict(actual_manufacturing_request=asdict(Request()),actual_blank_settings=asdict(Blank()),actual_relief_cad_sha256='a'*64,refinement_regions=[],
            requested_element_size_m=.001,refined_size_m=.0001,refinement_core_radius_m=.0003,refinement_radius_m=.0012,local_refinement_source_phase_rad=0.)
        mesh=dict(nodes_m=[[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]],tetrahedra=[[0,1,2,3]],
            boundary_triangles=[[0,2,1],[0,1,3],[0,3,2],[1,2,3]],boundary_tags=[1,1,1,1],metadata=metadata)
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)
            def save():
                for member in models:(folder/(member+'_mesh.json')).write_text(json.dumps(mesh),encoding='utf8')
            save();runner._load_meshes(folder,models,regions,cad,background_controls=background)
            metadata['refined_size_m']=.0002;save()
            with self.assertRaisesRegex(ValueError,'实际主体/背景场尺寸'):runner._load_meshes(folder,models,regions,cad,background_controls=background)
            # 未显式新policy时仍保持原loader默认兼容行为。
            runner._load_meshes(folder,models,regions,cad)
            metadata['refined_size_m']=.0001;metadata['actual_blank_settings']['bore_diameter_m']=.005;save()
            with self.assertRaisesRegex(ValueError,'制造'):runner._load_meshes(folder,models,regions,cad,background_controls=background)


if __name__=='__main__':unittest.main()
