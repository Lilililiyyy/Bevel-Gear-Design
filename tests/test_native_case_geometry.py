"""案例准备的0K身份／未知保留／mesh-only业务控制。"""
from copy import deepcopy
from dataclasses import dataclass
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from bevel_gear import native_case_geometry as geometry

PROJECT=Path(__file__).resolve().parents[1]
CASE=PROJECT/'examples/spiral_17_47_relieved_native_fe_case.json'


class GeometryPreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)

    def test_real_case_plan_is_lazy_and_reproduces_declared_grid_shapes(self):
        script="import importlib.util,sys,json;s=importlib.util.spec_from_file_location('standalone',sys.argv[1]);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);d=m.case_plan(m.load_case(sys.argv[2]));assert not any(x in sys.modules for x in ('numpy','scipy','gmsh','OCP'));print(json.dumps([(r['level'],[(p['grid']['x_count'],p['grid']['y_count']) for p in r['patches']]) for r in d['rows']]))"
        result=subprocess.run([sys.executable,'-S','-B','-c',script,str(Path(geometry.__file__)),str(CASE)],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr);rows=json.loads(result.stdout)
        self.assertEqual([r[1][0] for r in rows],[[85,83],[114,111],[170,166]])

    def test_regions_scope_rejects_other_phase_without_mesh_or_K(self):
        sys.path.insert(0,str(PROJECT/'examples'))
        from run_distributed_fe_contact import _load_regions_file
        loaded=geometry.load_case(CASE)
        document,*_=_load_regions_file(loaded['regions_file'],[0.],.001)
        self.assertEqual(len(document['regions']['pinion']),2)
        with self.assertRaisesRegex(ValueError,'所选相位'):_load_regions_file(loaded['regions_file'],[.01],.001)

    def test_repartition_covers_original_window_without_mask_input(self):
        loaded=geometry.load_case(CASE);old=loaded['case']['phases'][0]['patches'][0]['grid']
        new,proof=geometry.repartition_grid(old,225e-6,37.5e-6)
        self.assertEqual((new['x_count'],new['y_count']),(114,111))
        self.assertTrue(all(a>=b-1e-14 for a,b in zip(proof['new_cell_window_m'],proof['original_cell_window_m'])))
        self.assertNotIn('requested_mask',new)

    def test_old_pressure_fields_rejected_by_case_before_engines(self):
        for key in ('pressure_pa','gaps_m','C','accepted'):
            data=geometry.read_json(CASE);data['phases'][0][key]=1
            file=self.root/(key+'.json')
            data['base_case']='base.json';shutil.copyfile(PROJECT/'examples/spiral_17_47_relieved_contact_design.json',self.root/'base.json')
            geometry.write_json(file,data)
            with self.assertRaisesRegex(ValueError,'每相仅允许'):geometry.load_case(file)

    def test_unresolved_query_is_preserved_and_never_exported_as_template(self):
        loaded=geometry.load_case(CASE);loaded=deepcopy(loaded)
        loaded['case']['phases'][0]['patches']=[p for p in loaded['case']['phases'][0]['patches'] if p['tooth_offset']==0]
        loaded['tooth_offsets']=[0]
        class Query:
            def query(self,k,points):
                n=len(points);valid=np.ones(n,dtype=bool);unresolved=np.zeros(n,dtype=bool);status=np.full(n,'valid',dtype=object)
                valid[0]=False;unresolved[0]=True;status[0]='ray_newton_unresolved'
                return dict(valid=valid,torque=np.ones(n,dtype=bool),unresolved=unresolved,status=status)
        output=self.root/'unknown'
        with self.assertRaisesRegex(ArithmeticError,'未决点'):
            geometry.build_geometry_templates(loaded,SimpleNamespace(),output,query_factory=lambda phase:Query())
        self.assertFalse((output/'manifest.json').exists());records=list((output/'unresolved').rglob('*.json'))
        self.assertEqual(len(records),1);self.assertTrue(geometry.read_json(records[0])['unknown_is_not_OUT'])

    def test_mesh_only_uses_Gmsh_contract_and_never_assembles_K(self):
        from bevel_gear.elasticity import TetraMesh
        @dataclass
        class Request:tooth_count:int
        @dataclass
        class Settings:bore_diameter_m:float
        models={m:SimpleNamespace(request=Request(z),settings=Settings(.012),shape=object(),diagnostics={}) for m,z in [('pinion',17),('wheel',47)]}
        loaded=geometry.load_case(CASE)
        regions=dict(manufacturing={m:dict(request=dict(tooth_count=model.request.tooth_count),blank=dict(bore_diameter_m=.012)) for m,model in models.items()},
            regions={m:[dict(refinement_polylines_m=[[[0.,0.,0.],[.001,0.,0.]]],refined_size_m=18.75e-6,refinement_core_radius_m=50e-6,refinement_radius_m=150e-6)] for m in models})
        def mesh(*args,**kwargs):
            return TetraMesh(nodes_m=[[0.,0.,0.],[.001,0.,0.],[0.,.001,0.],[0.,0.,.001]],tetrahedra=[[0,1,2,3]],metadata={})
        def exported(model,path):Path(path).write_bytes(b'explicit mock STEP; no engine execution')
        with patch('bevel_gear.generated_tca.prepare_generated_tca',return_value='TCA'),\
                patch('bevel_gear.generated_nodal_ltca.generated_refinement_tracks',return_value={'tracks':{m:[[[0.,0.,0.],[.001,0.,0.]]] for m in models}}),\
                patch('bevel_gear.cad_import.ImportedCadModel',side_effect=lambda shape,diagnostics:SimpleNamespace(shape=shape)),\
                patch('bevel_gear.cad_import.export_model_step',side_effect=exported),\
                patch('bevel_gear.elasticity.mesh_step_tetra',side_effect=mesh) as gmsh,\
                patch('bevel_gear.elasticity.ElasticSystem',side_effect=AssertionError('mesh-only不能组K')),\
                patch('bevel_gear.generated_ltca.prepare_generated_ltca',side_effect=AssertionError('mesh-only不能prepare弹性')):
            cad={m:self.root/(m+'.brep') for m in models}
            for file in cad.values():file.write_bytes(b'explicit mock BRep; only binding contract, no CAD proof')
            result=geometry.generate_case_meshes(loaded,models,regions,self.root/'mesh',cad_brep_files=cad)
        self.assertEqual(set(result),{'pinion','wheel'});self.assertEqual(gmsh.call_count,2)
        self.assertEqual(gmsh.call_args.kwargs['refinement_regions'],regions['regions']['wheel'])
        self.assertFalse(geometry.read_json(self.root/'mesh/mesh_manifest.json')['K_assembled'])
        for m,body in result.items():self.assertEqual(body.metadata['actual_relief_cad_sha256'],geometry.file_sha256(cad[m]))


# 已核三档fresh纯CAD几何摘要；不含旧mask、gap、压力、W或C。
RECORDED_K2_GEOMETRY = json.loads(r'''[{"name":"grid300x50um","nx":85,"ny":83,"evidence":{"total_cells":7055,"requested_cells":0,"all_center_and_corner_positions_requeried":true,"torque_direction_rejected_centers":0,"incomplete_CAD_corner_cells":7055,"corners":{"reason_counts":{"opposing_normal_limit":50,"proven_outside_positive_NURBS_control_hull":7151,"wheel:outside_trimmed_face":23},"unresolved_count":0,"unresolved":[]},"centers":{"reason_counts":{"opposing_normal_limit":48,"pinion:outside_trimmed_face":2,"proven_outside_positive_NURBS_control_hull":6983,"wheel:outside_trimmed_face":22},"unresolved_count":0,"unresolved":[]},"completed":true}},{"name":"grid225x37p5um","nx":114,"ny":111,"evidence":{"total_cells":12654,"requested_cells":0,"all_center_and_corner_positions_requeried":true,"torque_direction_rejected_centers":0,"incomplete_CAD_corner_cells":12654,"corners":{"reason_counts":{"opposing_normal_limit":87,"pinion:outside_trimmed_face":3,"proven_outside_positive_NURBS_control_hull":12748,"wheel:OCP_proven_outside_trimmed_face":2,"wheel:outside_trimmed_face":40},"unresolved_count":0,"unresolved":[]},"centers":{"reason_counts":{"opposing_normal_limit":87,"pinion:outside_trimmed_face":2,"proven_outside_positive_NURBS_control_hull":12525,"wheel:OCP_proven_outside_trimmed_face":1,"wheel:outside_trimmed_face":39},"unresolved_count":0,"unresolved":[]},"completed":true}},{"name":"grid150x25um","nx":170,"ny":166,"evidence":{"total_cells":28220,"requested_cells":0,"all_center_and_corner_positions_requeried":true,"torque_direction_rejected_centers":0,"incomplete_CAD_corner_cells":28220,"corners":{"reason_counts":{"opposing_normal_limit":195,"pinion:outside_trimmed_face":4,"proven_outside_positive_NURBS_control_hull":28267,"wheel:OCP_proven_outside_trimmed_face":1,"wheel:outside_trimmed_face":90},"unresolved_count":0,"unresolved":[]},"centers":{"reason_counts":{"opposing_normal_limit":187,"pinion:outside_trimmed_face":5,"proven_outside_positive_NURBS_control_hull":27942,"wheel:OCP_proven_outside_trimmed_face":2,"wheel:outside_trimmed_face":84},"unresolved_count":0,"unresolved":[]},"completed":true}}]''')
RECORDED_K2_SOURCE_SHA256 = 'de6912cd3de1c716ae62ddf33d141a47c35c5746d5e87bef0d49fb45766767c8'


class EmptyGeometryPatchTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)

    @staticmethod
    def query(mode):
        class Query:
            def query(self,k,points):
                n=len(points);valid=np.full(n,mode=='positive' or mode=='mixed' and k!=2,dtype=bool)
                status=np.where(valid,'valid','proven_outside_positive_NURBS_control_hull').astype(object)
                if mode=='slope':status[:]='pinion:halfspace_slope_limit'
                return dict(valid=valid,torque=valid.copy(),unresolved=np.zeros(n,dtype=bool),status=status)
        return Query()

    def test_recorded_real_k2_three_fresh_CAD_grids_allow_explicit_empty_only(self):
        for row in RECORDED_K2_GEOMETRY:
            grid=dict(x_count=row['nx'],y_count=row['ny'],spacing_x_m=1e-4,spacing_y_m=1e-4,origin_xy_m=[0.,0.])
            proof=geometry.empty_geometry_patch_evidence(row['evidence'],grid)
            self.assertTrue(proof['empty_patch_retained']);self.assertFalse(proof['pressure_zero_certified'])
            self.assertFalse(proof['material_domain_certified'])
            damaged=deepcopy(row['evidence']);damaged['centers']['unresolved_count']=1
            with self.assertRaises(ArithmeticError):geometry.empty_geometry_patch_evidence(damaged,grid)

    def test_full_positive_and_mixed_query_keep_all_original_five_patches(self):
        loaded=geometry.load_case(CASE)
        for mode in ('positive','mixed'):
            output=self.root/mode
            manifest=geometry.build_geometry_templates(loaded,SimpleNamespace(),output,query_factory=lambda p:self.query(mode))
            self.assertEqual(len(manifest['entries']),3)
            for entry in manifest['entries']:
                patches=geometry.read_json(output/entry['file'])['template']['patches']
                self.assertEqual([p['tooth_offset'] for p in patches],loaded['tooth_offsets'])
                self.assertEqual(len(patches),5)
                for patch in patches:
                    mask=np.asarray(patch['requested_mask'])
                    self.assertEqual(bool(mask.any()),not(mode=='mixed' and patch['tooth_offset']==2))
                checks=geometry.read_json((output/entry['file']).with_suffix('.geometry_checks.json'))
                empty=next(p for p in checks['patches'] if p['tooth_offset']==2)['empty_geometry_patch']
                if mode=='mixed':self.assertTrue(empty['empty_patch_retained']);self.assertFalse(empty['pressure_zero_certified'])
                else:self.assertIsNone(empty)

    def test_all_empty_query_rejected_globally_with_original_offsets_retained(self):
        loaded=geometry.load_case(CASE);output=self.root/'empty'
        with self.assertRaisesRegex(ArithmeticError,'合计请求为空'):
            geometry.build_geometry_templates(loaded,SimpleNamespace(),output,query_factory=lambda p:self.query('empty'))
        self.assertFalse((output/'manifest.json').exists())
        saved=list((output/'rejected_all_empty').rglob('*.json'));self.assertEqual(len(saved),1)
        record=geometry.read_json(saved[0]);self.assertEqual(record['original_tooth_offsets'],loaded['tooth_offsets'])
        self.assertEqual(len(record['patches']),5);self.assertFalse(record['pressure_zero_certified'])

    def test_slope_model_exclusion_cannot_certify_empty_geometry(self):
        loaded=geometry.load_case(CASE);output=self.root/'slope'
        with self.assertRaisesRegex(ArithmeticError,'模型斜率限制'):
            geometry.build_geometry_templates(loaded,SimpleNamespace(),output,query_factory=lambda p:self.query('slope'))
        self.assertFalse((output/'manifest.json').exists())


if __name__=='__main__':unittest.main()
