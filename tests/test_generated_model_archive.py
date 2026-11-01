"""档案原字节／安全路径／制造绑定控制；mock字节不作实际CAD几何证明。"""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from bevel_gear import generated_model_archive as archive

PROJECT=Path(__file__).resolve().parents[1]


class GeneratedModelArchiveControls(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        base=json.loads((PROJECT/'examples/spiral_17_47_relieved_contact_design.json').read_text(encoding='utf8'))
        self.manufacturing={m:archive._manufacturing(base['manufacturing'][m]) for m in archive.MEMBERS}
        self.bindings={}
        for m in archive.MEMBERS:
            (self.root/(m+'.brep')).write_bytes(('explicit mock '+m+' full BRep; no actual CAD').encode())
            (self.root/(m+'_blank.brep')).write_bytes(('explicit mock '+m+' blank BRep; no actual CAD').encode())
            req=self.manufacturing[m]['request'];grid=dict(inputs=req,units='m',angle_units='rad',patches={n:dict(
                width_count=4,profile_count=4,points_m=[[float(i),float(j),0.] for j in range(4) for i in range(4)]) for n in archive.PATCHES})
            groups={n:[i] for i,n in enumerate(archive.PATCHES)}
            record=dict(schema='bevel_generated_model_archive_v1',member=m,units='m',angle_units='rad',cad_internal_length_unit='mm',
                request=req,settings=self.manufacturing[m]['blank'],grid=grid,diagnostics=dict(faces=6,solids=1),
                blank_diagnostics=dict(faces=3,solids=1),blank_profile_m=[[.01,0.],[.02,0.],[.02,.01]],
                face_groups=dict(groups,blank=[5]),slot_face_groups={str(i):groups for i in range(req['tooth_count'])},
                shell_sewing_tolerance_m=1e-9,capabilities=archive.CAPABILITIES,
                files={kind:dict(file=file,sha256=archive._sha(self.root/file)) for kind,file in
                    [('gear_brep',m+'.brep'),('blank_brep',m+'_blank.brep')]})
            path=self.root/(m+'_model.json');path.write_text(json.dumps(record,ensure_ascii=False,allow_nan=False),encoding='utf8')
            self.bindings[m]=archive._validated(path,expected_descriptor_sha256=archive._sha(path),expected_member=m,
                expected_manufacturing=self.manufacturing[m],expected_gear_sha256=record['files']['gear_brep']['sha256'])[2]

    def options(self,m='pinion'):
        b=self.bindings[m]
        return dict(expected_descriptor_sha256=b['descriptor_sha256'],expected_member=m,expected_manufacturing=self.manufacturing[m],expected_gear_sha256=b['gear_brep_sha256'])

    def changed(self,update):
        path=self.root/'pinion_model.json';record=json.loads(path.read_text(encoding='utf8'));update(record)
        path.write_text(json.dumps(record,ensure_ascii=False,allow_nan=False),encoding='utf8');options=self.options();options['expected_descriptor_sha256']=archive._sha(path)
        return path,options

    def test_lazy_standalone_import_needs_no_optional_dependencies(self):
        code="import importlib.util,sys;s=importlib.util.spec_from_file_location('single',sys.argv[1]);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);assert not any(x in sys.modules for x in ('numpy','scipy','OCP'))"
        result=subprocess.run([sys.executable,'-S','-B','-c',code,str(Path(archive.__file__))],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_copy_keeps_complete_descriptor_and_both_dependencies_byte_identical(self):
        originals={p.name:p.read_bytes() for p in self.root.iterdir() if p.is_file()};destination=self.root/'copied'
        for m in archive.MEMBERS:
            binding=archive.copy_generated_model_archive(self.root/(m+'_model.json'),destination,**self.options(m))
            self.assertEqual(binding,self.bindings[m]);self.assertFalse(archive._read(destination/binding['descriptor_file'])['capabilities']['slot_shapes_included'])
        for name,raw in originals.items():self.assertEqual((self.root/name).read_bytes(),raw);self.assertEqual((destination/name).read_bytes(),raw)

    def test_path_escape_basename_controls_before_optional_engine(self):
        original=(self.root/'pinion_model.json').read_bytes()
        for name in ('../escape.brep','sub/file.brep',chr(67)+':'+chr(92)+'escape.brep',chr(92)+'escape.brep'):
            (self.root/'pinion_model.json').write_bytes(original)
            path,options=self.changed(lambda r:r['files']['gear_brep'].update(file=name))
            with self.assertRaisesRegex(ValueError,'basename'):archive.load_generated_model_archive(path,**options)

    def test_wrong_CAD_wrong_manufacturing_and_original_file_tampering_rejected(self):
        path=self.root/'pinion_model.json';options=self.options();options['expected_gear_sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'CAD身份'):archive.load_generated_model_archive(path,**options)
        options=self.options();options['expected_manufacturing']=deepcopy(self.manufacturing['pinion']);options['expected_manufacturing']['blank']['bore_diameter_m']*=1.01
        with self.assertRaisesRegex(ValueError,'制造'):archive.load_generated_model_archive(path,**options)
        blank=self.root/'pinion_blank.brep';blank.write_bytes(blank.read_bytes()+b'changed')
        with self.assertRaisesRegex(ValueError,'BRep文件SHA'):archive.load_generated_model_archive(path,**self.options())

    def test_descriptor_grid_face_and_units_tampering_rejected(self):
        path=self.root/'pinion_model.json';original=path.read_bytes();path.write_bytes(original+b' ')
        with self.assertRaisesRegex(ValueError,'描述身份'):archive.load_generated_model_archive(path,**self.options())
        for change,error in [(lambda r:r['grid']['inputs'].update(tip_relief_m=0.),'grid'),
                (lambda r:r['face_groups'].update(left_flank=[999999]),'面号'),(lambda r:r.update(units='mm'),'单位'),
                (lambda r:r['capabilities'].update(slot_shapes_included=True),'能力')]:
            path.write_bytes(original);file,options=self.changed(change)
            with self.assertRaisesRegex(ValueError,error):archive.load_generated_model_archive(file,**options)

    def test_workflow_bindings_checked_before_either_model_load(self):
        inputs=dict(generated_model_archives=deepcopy(self.bindings),actual_manufacturing=self.manufacturing,
            actual_cad_sha256={m:b['gear_brep_sha256'] for m,b in self.bindings.items()},cad_files={m:b['gear_brep_file'] for m,b in self.bindings.items()})
        inputs['generated_model_archives']['wheel']['blank_brep_sha256']='0'*64
        with patch.object(archive,'load_generated_model_archive',side_effect=AssertionError('任一绑定错时不应先读另一模型')) as loaded:
            with self.assertRaisesRegex(ValueError,'依赖'):archive.load_native_model_archives(inputs,self.root)
        loaded.assert_not_called()

    def test_conflicting_copy_destination_never_overwrites_existing_bytes(self):
        destination=self.root/'conflict';destination.mkdir();target=destination/'pinion.brep';target.write_bytes(b'caller existing bytes')
        with self.assertRaises(FileExistsError):archive.copy_generated_model_archive(self.root/'pinion_model.json',destination,**self.options())
        self.assertEqual(target.read_bytes(),b'caller existing bytes')

    def test_producer_requires_complete_model_archive_for_real_CAD_only(self):
        spec=importlib.util.spec_from_file_location('model_archive_producer_control',PROJECT/'examples/run_native_fe_case.py')
        runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
        self.assertEqual(runner._model_archive_documents(dict(cad_binding_state='synthetic_FE_fixture_no_CAD'),self.root/'synthetic',None),{})
        inputs=dict(cad_binding_state='manufacturing_BRep_mesh_bound',actual_manufacturing=self.manufacturing,
            actual_cad_sha256={m:b['gear_brep_sha256'] for m,b in self.bindings.items()},cad_files={m:b['gear_brep_file'] for m,b in self.bindings.items()})
        with self.assertRaisesRegex(ValueError,'完整模型档案'):runner._model_archive_documents(inputs,self.root/'missing',None)
        sources={m:dict(descriptor_path=self.root/b['descriptor_file'],descriptor_sha256=b['descriptor_sha256']) for m,b in self.bindings.items()}
        output=runner._model_archive_documents(inputs,self.root/'level',sources)
        self.assertEqual(output['generated_model_archives'],self.bindings)


if __name__=='__main__':unittest.main()
