"""真实小TET多相／三格、正式档案、partial前拒和接触内存策略控制。"""
from copy import deepcopy
from dataclasses import asdict,dataclass,fields
import gzip
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from examples.validate_native_contact import build_fixture
from bevel_gear.native_case_geometry import repartition_grid
from bevel_gear.native_fe_contact import solve_native_fe_contact
from bevel_gear.native_fe_operator import load_native_operator
from bevel_gear.native_pressure_domain import audit_native_response_archive
from bevel_gear.native_fe_archive import save_native_fe_response

PROJECT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('native_case_runner_under_test',PROJECT/'examples/run_native_fe_case.py')
runner=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(runner)


def rotated_phase(template,angle):
    c,s=np.cos(angle),np.sin(angle);R=np.array([[c,0.,s],[0.,1.,0.],[-s,0.,c]])
    result=deepcopy(template);result['driving_angle_rad']=float(angle);result['reference_follower_angle_rad']=-float(angle)
    for m in ('pinion','wheel'):result['frames'][m+'_rotation_rows']=R.copy()
    for p in result['patches']:
        plane=p['plane'];plane['origin_world_m']=R@np.asarray(plane['origin_world_m'])
        plane['tangent_rows']=np.asarray(plane['tangent_rows'])@R.T;plane['normal_world']=R@np.asarray(plane['normal_world'])
        p['source_material_reference_frames']=deepcopy(result['frames'])
    return result


def study_inputs(angles):
    return dict(cad_binding_state='synthetic_FE_fixture_no_CAD',base_case=dict(manufacturing={m:dict(request=dict(tooth_count=17),blank={}) for m in ('pinion','wheel')},
        materials={m:dict(young_modulus_pa=210e9,poisson_ratio=.3,density_kg_m3=7850.,heat_treatment='generic') for m in ('pinion','wheel')},cad_sampling={}),
        effective=dict(phases=list(angles),selected=list(range(len(angles)))),
        distributed_fe_case=dict(schema='bevel_distributed_fe_case_v1',geometry_source='FE_boundary',analysis=dict(driver_angles_rad=list(angles))),
        selected_phase_indices=list(range(len(angles))))


def make_study_fixture(output):
    """只创建两块合计<1000节点的真实FE；可供私有reader交叉审核复用。"""
    prepared,template,compact=build_fixture();angles=(-.0003,.0003);entries=[]
    spacing=template['patches'][0]['grid']['spacing_x_m']
    for name,factor in (('coarse',1.),('middle',.8),('fine',2/3)):
        for i,angle in enumerate(angles):
            current=rotated_phase(template,angle)
            for p in current['patches']:
                p['grid']=repartition_grid(p['grid'],spacing*factor,spacing*factor)[0]
                p['requested_mask']=np.ones((p['grid']['y_count'],p['grid']['x_count']),dtype=bool)
            entries.append(dict(name=name,phase_index=i,template=current))
    inputs=study_inputs(angles)
    study=runner.run_grid_study(prepared,entries,output,workflow_inputs=inputs,compact_facets=compact,solver=dict(batch_cells=32))
    return prepared,study,inputs


class NativeCaseRealTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)

    def test_two_actual_phases_three_grids_use_same_prepared_and_formal_archive(self):
        # 制造刚度只允许fixture构造；runner期间不允许再prepare或组装K。
        with patch('bevel_gear.generated_ltca.prepare_generated_ltca',side_effect=AssertionError('runner不能再prepare K')):
            prepared,study,inputs=make_study_fixture(self.root/'study')
        self.assertTrue(study['analysis_completed']);self.assertEqual(len(study['levels']),3)
        self.assertTrue(study['same_prepared_used']);self.assertFalse(study['cross_grid_force_or_C_reused'])
        for level in study['levels']:
            folder=self.root/'study'/level['name'];summary=json.loads((folder/'distributed_fe_summary.json').read_text(encoding='utf8'))
            self.assertEqual(summary['selected_phase_indices'],[0,1]);self.assertTrue(summary['analysis_completed'])
            for row in summary['states']:
                phase=folder/row['native_phase_archive']['run_root']
                operator=load_native_operator(phase/'operator')
                with gzip.open(phase/'state.json.gz','rt',encoding='utf8') as stream:state=json.load(stream)
                result,proof=audit_native_response_archive(phase/'state.json.gz',run_root=phase,native_system=operator,
                    meshes={m:s.mesh for m,s in prepared.systems.items()},materials={m:dict(young_modulus_pa=210e9,poisson_ratio=.3) for m in prepared.systems},
                    fixed_nodes={m:np.unique(s.fixed_dofs//3) for m,s in prepared.systems.items()},mesh_source_sha256=summary['mesh_sha256'])
                self.assertTrue(proof['passed']);self.assertTrue(proof['independent_Ku_recheck'])
                self.assertNotEqual(state['input_fingerprint'],summary['input_fingerprint'])
                self.assertEqual(result['source_input_fingerprint'],state['input_fingerprint'])

    def test_partial_original_request_never_reaches_QP(self):
        prepared,template,compact=build_fixture();template['patches'][0]['grid']['origin_xy_m']=(-.0007,-.0004)
        with patch('bevel_gear.generated_distributed_fe.solve_distributed_contact_system',side_effect=AssertionError('partial不能进入QP')) as qp:
            study=runner.run_grid_study(prepared,[dict(name='one',phase_index=0,template=template)],self.root/'partial',
                workflow_inputs=study_inputs([0.]),compact_facets=compact)
        qp.assert_not_called();self.assertFalse(study['analysis_completed'])
        state=json.loads((self.root/'partial/one/states/phase_0000.json').read_text(encoding='utf8'))
        self.assertFalse(state['accepted']);self.assertNotIn('native_phase_archive',state)

    @unittest.skipUnless(importlib.util.find_spec('psutil') is not None,'接触专用内存计量需要可选psutil')
    def test_contact_only_same_K_and_raw_response_arrays_remain_bitwise_equal(self):
        from bevel_gear.native_contact_memory import release_prepared_assembly_caches
        prepared,template,compact=build_fixture();original=dict(prepared.systems)
        keys={m:s.stiffness for m,s in original.items()};factors={m:s.factorization for m,s in original.items()}
        first=solve_native_fe_contact(prepared,template,compact_facets=compact,include_body_fields=True,batch_cells=32)
        self.assertTrue(first['accepted'])
        save_native_fe_response(prepared,first['system'],first['result'],self.root/'full',phase_index=0,input_fingerprint='a'*64)
        released=release_prepared_assembly_caches(prepared)
        self.assertTrue(released['complete_cache_release']);self.assertFalse(released['element_stress_available'])
        second=solve_native_fe_contact(prepared,template,compact_facets=compact,include_body_fields=True,batch_cells=32)
        self.assertTrue(second['accepted']);self.assertEqual(first['input_fingerprint'],second['input_fingerprint'])
        save_native_fe_response(prepared,second['system'],second['result'],self.root/'contact_only',phase_index=0,input_fingerprint='a'*64)
        count=0
        for file in (self.root/'full').rglob('*.npz'):
            other=self.root/'contact_only'/file.relative_to(self.root/'full')
            with np.load(file,allow_pickle=False) as a,np.load(other,allow_pickle=False) as b:
                self.assertEqual(a.files,b.files)
                for name in a.files:
                    self.assertEqual(a[name].dtype,b[name].dtype);self.assertEqual(a[name].shape,b[name].shape)
                    self.assertEqual(a[name].tobytes(),b[name].tobytes());count+=1
        self.assertGreaterEqual(count,22)
        for m,s in prepared.systems.items():
            self.assertIs(s.stiffness,keys[m]);self.assertIs(s.factorization,factors[m])
            with self.assertRaisesRegex(NotImplementedError,'单元应力'):s.solve(np.zeros(s.stiffness.shape[0]))
            with self.assertRaises(NotImplementedError):original[m].solve(np.zeros(s.stiffness.shape[0]))

    def test_old_physics_fields_rejected_before_native_QP(self):
        prepared,template,compact=build_fixture();template['pressure_pa']=[1e9]
        with patch('bevel_gear.native_fe_contact.solve_native_fe_contact',side_effect=AssertionError('不得读旧压力')) as solve:
            study=runner.run_grid_study(prepared,[dict(name='one',phase_index=0,template=template)],self.root/'forbidden',workflow_inputs=study_inputs([0.]),compact_facets=compact)
        solve.assert_not_called();self.assertFalse(study['analysis_completed'])

    def test_CAD_file_request_blank_and_mesh_SHA_binding_rejects_each_difference(self):
        # 仅验证档案同源业务协议；这些显式mock字节不构成有效CAD几何证据。
        from bevel_gear.native_case_geometry import read_json,normalized_manufacturing,file_sha256
        base=read_json(PROJECT/'examples/spiral_17_47_relieved_contact_design.json');actual=normalized_manufacturing(base)
        files={m:self.root/(m+'.brep') for m in actual}
        for m,file in files.items():file.write_bytes(('explicit mock '+m+' BRep bytes; no CAD engine').encode())
        digests={m:file_sha256(p) for m,p in files.items()}
        inputs=dict(base_case=base,actual_manufacturing=actual,actual_cad_sha256=digests)
        prepared=SimpleNamespace(systems={m:SimpleNamespace(mesh=SimpleNamespace(metadata=dict(
            actual_manufacturing_request=deepcopy(actual[m]['request']),actual_blank_settings=deepcopy(actual[m]['blank']),
            actual_relief_cad_sha256=digests[m]))) for m in actual})
        destination=self.root/'good';destination.mkdir()
        binding=runner._cad_documents(prepared,inputs,destination,files)
        self.assertEqual(binding['cad_files'],{m:m+'.brep' for m in actual})
        for member,field in [('pinion','actual_manufacturing_request'),('wheel','actual_blank_settings'),('wheel','actual_relief_cad_sha256')]:
            damaged=deepcopy(prepared);damaged.systems[member].mesh.metadata[field]='wrong'
            folder=self.root/(member+field);folder.mkdir()
            with self.assertRaisesRegex(ValueError,'绑定身份'):runner._cad_documents(damaged,inputs,folder,files)
        damaged=deepcopy(inputs);damaged['base_case']['manufacturing']['pinion']['blank']['support_depth_factor']=3.
        folder=self.root/'raw_changed';folder.mkdir()
        with self.assertRaisesRegex(ValueError,'raw制造'):runner._cad_documents(prepared,damaged,folder,files)
        files['wheel'].write_bytes(b'changed original file')
        folder=self.root/'CAD_changed';folder.mkdir()
        with self.assertRaisesRegex(ValueError,'绑定身份'):runner._cad_documents(prepared,inputs,folder,files)

    def test_factor_telemetry_reads_iparm_without_changing_real_small_solve(self):
        prepared,_,_=build_fixture();factor=prepared.systems['pinion'].factorization;n=len(prepared.systems['pinion'].free_dofs)
        original=np.arange(64,dtype=np.int64)
        class Proxy:
            iparm=original
            def solve(self,B):return factor.solve(B)
        wrapped=runner._TelemetryFactor(Proxy(),'pinion',self.root/'calls.jsonl');B=np.ones((n,2))
        np.testing.assert_array_equal(wrapped.solve(B),factor.solve(B));np.testing.assert_array_equal(original,np.arange(64))
        row=json.loads((self.root/'calls.jsonl').read_text(encoding='utf8'))
        self.assertEqual(row['iparm_zero_based'],dict(actual_refinement_count=6,refinement_limit=7,backward_tolerance=8,perturbed_pivots=13))
        self.assertEqual(row['rhs_shape'],[n,2]);self.assertIsNone(row['failure'])

    def _serialized_real_TET_loader_fixture(self,directory):
        """仅合成小TET的文件／类型合同；显式mock CAD摘要不作几何同源证明。"""
        from bevel_gear.native_case_geometry import write_json
        prepared,template,compact=build_fixture();directory.mkdir()
        @dataclass
        class Request:tooth_count:int=17
        @dataclass
        class Blank:bore_diameter_m:float=.012
        models={m:SimpleNamespace(request=Request(),settings=Blank()) for m in prepared.systems}
        cad={m:'a'*64 for m in prepared.systems};regions=dict(regions={m:[] for m in prepared.systems});before={}
        for m,body in prepared.systems.items():
            body.mesh.metadata.update(actual_manufacturing_request=asdict(models[m].request),actual_blank_settings=asdict(models[m].settings),
                actual_relief_cad_sha256=cad[m],refinement_regions=[])
            target=directory/(m+'_mesh.json');write_json(target,{f.name:getattr(body.mesh,f.name) for f in fields(body.mesh)});before[m]=target.read_bytes()
        return prepared,template,compact,models,regions,cad,before

    def test_loaded_arrays_preserve_raw_K_input_bytes_and_formal_response_archive(self):
        from bevel_gear.elasticity import ElasticSystem
        prepared,template,compact,models,regions,cad,before=self._serialized_real_TET_loader_fixture(self.root/'meshes')
        loaded=runner._load_meshes(self.root/'meshes',models,regions,cad);systems={}
        for m,mesh in loaded.items():
            original=prepared.systems[m]
            for name in ('nodes_m','tetrahedra','boundary_triangles','boundary_tags'):
                value=getattr(mesh,name);self.assertIsInstance(value,np.ndarray)
                self.assertEqual(value.dtype,np.dtype('float64' if name=='nodes_m' else 'int64'))
                self.assertEqual(value.tobytes(),getattr(original.mesh,name).tobytes())
            self.assertEqual(mesh.metadata,original.mesh.metadata)
            systems[m]=ElasticSystem(mesh,original.material,np.unique(original.fixed_dofs//3),maximum_dofs=1200)
            for name in ('data','indices','indptr'):
                self.assertEqual(getattr(systems[m].stiffness,name).tobytes(),getattr(original.stiffness,name).tobytes())
        normalized=SimpleNamespace(systems=systems,models=prepared.models)
        first=solve_native_fe_contact(prepared,template,compact_facets=compact,include_body_fields=True,batch_cells=32)
        second=solve_native_fe_contact(normalized,template,compact_facets=compact,include_body_fields=True,batch_cells=32)
        self.assertTrue(first['accepted']);self.assertTrue(second['accepted']);self.assertEqual(first['input_fingerprint'],second['input_fingerprint'])
        for folder,p,answer in [('original',prepared,first),('normalized',normalized,second)]:
            save_native_fe_response(p,answer['system'],answer['result'],self.root/folder,phase_index=0,input_fingerprint='b'*64)
        count=0
        for file in (self.root/'original').rglob('*.npz'):
            with np.load(file,allow_pickle=False) as a,np.load(self.root/'normalized'/file.relative_to(self.root/'original'),allow_pickle=False) as b:
                self.assertEqual(a.files,b.files)
                for key in a.files:
                    self.assertEqual(a[key].dtype,b[key].dtype);self.assertEqual(a[key].shape,b[key].shape);self.assertEqual(a[key].tobytes(),b[key].tobytes());count+=1
        self.assertGreaterEqual(count,22)
        study=runner.run_grid_study(normalized,[dict(name='one',phase_index=0,template=template)],self.root/'study',workflow_inputs=study_inputs([0.]),
            compact_facets=compact,solver=dict(batch_cells=32),source_mesh_files={m:self.root/'meshes'/(m+'_mesh.json') for m in systems})
        self.assertTrue(study['analysis_completed']);phase=self.root/'study/one/native_responses/phase_0000'
        summary=json.loads((self.root/'study/one/distributed_fe_summary.json').read_text(encoding='utf8'))
        _,proof=audit_native_response_archive(phase/'state.json.gz',run_root=phase,native_system=load_native_operator(phase/'operator'),
            meshes=loaded,materials={m:dict(young_modulus_pa=210e9,poisson_ratio=.3) for m in systems},
            fixed_nodes={m:np.unique(s.fixed_dofs//3) for m,s in systems.items()},mesh_source_sha256=summary['mesh_sha256'])
        self.assertTrue(proof['passed']);self.assertTrue(proof['independent_Ku_recheck'])
        for m,raw in before.items():self.assertEqual((self.root/'meshes'/(m+'_mesh.json')).read_bytes(),raw)

    def test_loader_refuses_real_or_overflow_topology_instead_of_truncating(self):
        from bevel_gear.native_case_geometry import write_json
        _,_,_,models,regions,cad,before=self._serialized_real_TET_loader_fixture(self.root/'source')
        for label,value in [('float',1.),('overflow',2**63)]:
            directory=self.root/label;directory.mkdir()
            for m,raw in before.items():
                document=json.loads(raw)
                if m=='pinion':document['tetrahedra'][0][0]=value
                write_json(directory/(m+'_mesh.json'),document)
            with self.assertRaisesRegex(ValueError,'不能截断'):runner._load_meshes(directory,models,regions,cad)


if __name__=='__main__':unittest.main()
