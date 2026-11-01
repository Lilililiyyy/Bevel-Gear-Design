"""真实小TET4保存→最终operator重载→独立Ku/g/W审核及篡改拒绝。"""
from dataclasses import asdict
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from examples.validate_native_contact import build_fixture
from bevel_gear.native_fe_contact import solve_native_fe_contact
from bevel_gear.native_fe_operator import save_native_operator,load_native_operator,safe
from bevel_gear.native_fe_archive import (save_native_fe_response,save_native_fe_phase_state,
    native_phase_input_fingerprint,file_sha256)
from bevel_gear.native_pressure_domain import audit_native_response_archive


def write_json(path,value):
    path.write_text(json.dumps(safe(value),ensure_ascii=False,sort_keys=True,separators=(',',':'))+'\n',encoding='utf8')


class NativePhaseArchiveRealTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prepared,cls.template,cls.compact=build_fixture()
        cls.answer=solve_native_fe_contact(cls.prepared,cls.template,compact_facets=cls.compact,
            include_body_fields=True,batch_cells=32)
        if not cls.answer['accepted']:raise AssertionError(cls.answer['rejection_reasons'])

    def fixture(self,root):
        folder=root/'phase_0003';folder.mkdir();mesh_folder=root/'meshes';mesh_folder.mkdir()
        meshes={m:s.mesh for m,s in self.prepared.systems.items()}
        mesh_sha={}
        for m,mesh in meshes.items():
            file=mesh_folder/(m+'_mesh.json');write_json(file,asdict(mesh));mesh_sha[m]=file_sha256(file)
        system=dict(self.answer['system'])
        operator=save_native_operator(system,folder/'operator',actual_mesh_sha256=mesh_sha)
        core=dict(self.answer['result'],**operator,actual_native_mesh_sha256=mesh_sha)
        api_fp=self.answer['input_fingerprint'];global_fp=hashlib.sha256(b'explicit synthetic global workflow').hexdigest()
        inputs=dict(schema='bevel_native_FE_phase_inputs_v1',phase_index=3,driving_angle_rad=0.,
            global_workflow_input_fingerprint=global_fp,native_contact_api_input_fingerprint=api_fp,
            actual_native_mesh_sha256=mesh_sha,geometry_template=self.template,
            materials={m:dict(young_modulus_pa=210e9,poisson_ratio=.3) for m in meshes},
            final_operator_binding=dict(operator,current_operator_manifest_file='operator/manifest.json'))
        fp=native_phase_input_fingerprint(inputs)
        receipt=save_native_fe_response(self.prepared,system,core,folder,phase_index=3,input_fingerprint=fp)
        identity={'native_fe_contact.py':file_sha256(Path(__file__).resolve().parents[1]/'bevel_gear/native_fe_contact.py')}
        arguments=dict(inputs=inputs,result=core,response_receipt=receipt,phase_index=3,driving_angle_rad=0.,
            source_identity_before=identity,source_identity_after=identity,
            native_contact_api_input_fingerprint=api_fp,global_workflow_input_fingerprint=global_fp)
        archived=save_native_fe_phase_state(folder,**arguments)
        loaded=load_native_operator(folder/'operator')
        kwargs=dict(run_root=folder,native_system=loaded,meshes=meshes,
            materials=inputs['materials'],mesh_source_sha256=mesh_sha,
            fixed_nodes={m:np.unique(s.fixed_dofs//3) for m,s in self.prepared.systems.items()})
        return folder,archived,kwargs,arguments

    def test_real_nonzero_contact_roundtrip_independent_Ku_gap_and_W(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder,reference,kwargs,args=self.fixture(Path(temporary))
            state=folder/reference['state_file']
            result,proof=audit_native_response_archive(state,**kwargs)
            self.assertTrue(proof['passed'],proof)
            self.assertTrue(proof['independent_Ku_recheck']);self.assertFalse(proof['NURBS_gap_queried'])
            self.assertTrue(proof['original_state_unchanged'])
            self.assertEqual(result['source_input_fingerprint'],reference['input_fingerprint'])
            self.assertNotEqual(reference['input_fingerprint'],self.answer['input_fingerprint'])
            self.assertEqual(result['native_contact_api_input_fingerprint'],self.answer['input_fingerprint'])
            self.assertEqual(result['current_operator_manifest_file'],'operator/manifest.json')
            self.assertEqual(Path(kwargs['native_system']['current_operator_manifest_file']).resolve(),(folder/'operator/manifest.json').resolve())
            self.assertEqual(result['active_cell_count'],4)
            self.assertNotEqual(kwargs['native_system']['geometry']['reference_follower_angle_rad'],0.)
            self.assertEqual(kwargs['native_system']['geometry']['reference_follower_angle_rad'],
                self.answer['result']['reference_follower_angle_rad'])
            self.assertNotIn('input_fingerprint',json.loads((folder/'inputs.json').read_text(encoding='utf8')))
            self.assertNotIn('body_fields',json.loads(gzip.decompress(state.read_bytes()))['result'])
            self.assertIn('body_fields',self.answer['result'])
            with self.assertRaises(FileExistsError):save_native_fe_phase_state(folder,**args)

    def test_actual_phase_inputs_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder,reference,kwargs,_=self.fixture(Path(temporary))
            file=folder/'inputs.json';inputs=json.loads(file.read_text(encoding='utf8'));inputs['materials']['pinion']['young_modulus_pa']*=1.01
            write_json(file,inputs)
            with self.assertRaisesRegex(ValueError,'输入JSON.*fingerprint'):
                audit_native_response_archive(folder/reference['state_file'],**kwargs)

    def test_final_and_response_receipt_tamper_rejected(self):
        for kind in ('final','response'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as temporary:
                folder,reference,kwargs,_=self.fixture(Path(temporary))
                file=folder/('final_state_receipt.json' if kind=='final' else 'response_receipt.json')
                data=json.loads(file.read_text(encoding='utf8'))
                if kind=='final':data['source_state_sha256']='0'*64
                else:data['bodies']['pinion']['response_sha256']='0'*64
                write_json(file,data)
                with self.assertRaisesRegex(ValueError,'receipt'):
                    audit_native_response_archive(folder/reference['state_file'],**kwargs)

    def test_relative_manifest_path_traversal_rejected_even_with_state_rehashed(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder,reference,kwargs,_=self.fixture(Path(temporary))
            state_file=folder/reference['state_file'];state=json.loads(gzip.decompress(state_file.read_bytes()))
            state['result']['current_operator_manifest_file']='../outside/manifest.json'
            with gzip.open(state_file,'wt',encoding='utf8') as stream:json.dump(state,stream)
            final_file=folder/'final_state_receipt.json';final=json.loads(final_file.read_text(encoding='utf8'))
            final['source_state_sha256']=file_sha256(state_file);write_json(final_file,final)
            with self.assertRaises(ValueError):audit_native_response_archive(state_file,**kwargs)

    def test_self_fingerprint_and_nonfinite_inputs_are_refused(self):
        with self.assertRaisesRegex(ValueError,'自身输入指纹'):
            native_phase_input_fingerprint(dict(input_fingerprint='a'*64))
        with self.assertRaises(ValueError):native_phase_input_fingerprint(dict(torque_nm=float('nan')))

    def test_partial_document_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder,_,_,arguments=self.fixture(Path(temporary))
            # 仅在本测试临时档案中制造首文件写出中断，保留原临时字节。
            for name in ('inputs.json','response_receipt.json','state.json.gz','final_state_receipt.json'):
                (folder/name).unlink()
            partial=folder/'inputs.json.tmp';original=b'original interrupted document'
            partial.write_bytes(original)
            with self.assertRaises(FileExistsError):save_native_fe_phase_state(folder,**arguments)
            self.assertEqual(partial.read_bytes(),original)


if __name__=='__main__':unittest.main(verbosity=2)
