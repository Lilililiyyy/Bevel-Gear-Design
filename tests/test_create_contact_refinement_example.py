"""窄带规划例程的冷启动协议/失败保存；CAD与几何生成mock，无大计算。"""
from contextlib import redirect_stdout
import importlib.util,io,json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch


class ContactRefinementExampleTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        project=next(p for p in Path(__file__).resolve().parents if (p/'examples/create_contact_refinement.py').is_file())
        spec=importlib.util.spec_from_file_location('contact_refinement_example',project/'examples/create_contact_refinement.py')
        self.module=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.module)
        self.base=self.root/'case.json';self.output=self.root/'regions.json'
        self.base.write_text(json.dumps(dict(schema='bevel_contact_manufacturing_case_v1',
            manufacturing={m:dict(request=dict(tooth_count=z),blank=dict(bore_diameter_m=.001)) for m,z in [('pinion',17),('wheel',47)]})),encoding='utf-8')

    def invoke(self,*,failure=None):
        generated=dict(document=dict(schema='bevel_refinement_regions_v1',units='m'),
            diagnostics=dict(all_requested_phases_generated=True,pressure_solved=False,track_counts=dict(pinion=1,wheel=1)))
        with patch('bevel_gear.generated_solid.build_generated_gear',return_value=object()) as cad,\
             patch('bevel_gear.generated_halfspace.prepare_generated_halfspace',return_value=object()),\
             patch('bevel_gear.contact_refinement.create_contact_refinement',return_value=generated,side_effect=failure) as geometry,redirect_stdout(io.StringIO()):
            code=self.module.main([str(self.base),'--output',str(self.output),'--intervals','8'])
            return code,cad,geometry

    def test_default_nine_phase_plan_does_not_require_materials_or_mechanical_load(self):
        code,cad,geometry=self.invoke();self.assertEqual(code,0);self.assertEqual(cad.call_count,2)
        self.assertEqual(len(geometry.call_args.kwargs['driving_angles_rad']),9)
        self.assertEqual(geometry.call_args.kwargs['tooth_offsets'],[-1,0,1])
        self.assertTrue(self.output.is_file())
        diagnostic=json.loads(self.output.with_name('regions.diagnostics.json').read_text(encoding='utf-8'))
        self.assertTrue(diagnostic['source_identity_unchanged']);self.assertFalse(diagnostic['pressure_solved'])

    def test_failed_geometry_does_not_export_a_complete_regions_file(self):
        with self.assertRaises(ArithmeticError):self.invoke(failure=ArithmeticError('真实面采样未完成'))
        self.assertFalse(self.output.exists())
        diagnostic=json.loads(self.output.with_name('regions.diagnostics.json').read_text(encoding='utf-8'))
        self.assertFalse(diagnostic['all_requested_phases_generated'])


if __name__=='__main__':unittest.main()
