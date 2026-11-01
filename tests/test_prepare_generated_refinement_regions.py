"""按轮附加加密区的prepare编排；mock CAD/网格/线弹性，不冒充物理验证。"""
from contextlib import ExitStack
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch


@unittest.skipUnless(importlib.util.find_spec('numpy') and importlib.util.find_spec('scipy'),'需要NumPy/SciPy')
class PrepareGeneratedRefinementRegionTests(unittest.TestCase):
    def prepared(self,regions):
        import numpy as np
        from bevel_gear.generated_ltca import prepare_generated_ltca
        from bevel_gear.generated_solid import GeneratedBlankSettings
        from bevel_gear.elasticity import TetraMesh
        requests={m:SimpleNamespace(tooth_count=z) for m,z in [('pinion',17),('wheel',47)]}
        settings=GeneratedBlankSettings(bore_diameter_m=.002)
        models={m:SimpleNamespace(request=request,settings=settings,shape=object(),
            diagnostics=dict(solids=1,valid=True,object_scope='full_gear',volume_m3=1e-6)) for m,request in requests.items()}
        mesh=TetraMesh([[.001,0.,0.],[0.,.001,0.],[-.001,0.,0.],[0.,0.,.001]],[[0,1,2,3]],metadata=dict(units='m'))
        reference=dict(converged=True,frames={m+'_rotation_rows':np.eye(3).tolist() for m in models},
            **{m+'_contact_point_world_m':[0.,0.,0.] for m in models})
        with tempfile.TemporaryDirectory() as directory,ExitStack() as stack:
            stack.enter_context(patch('bevel_gear.generated_surface._request',side_effect=lambda value:value))
            stack.enter_context(patch('bevel_gear.generated_ltca.prepare_generated_tca',return_value=object()))
            stack.enter_context(patch('bevel_gear.generated_ltca.solve_generated_tca',return_value=reference))
            stack.enter_context(patch('bevel_gear.generated_ltca._check_mesh_metadata',return_value={}))
            stack.enter_context(patch('bevel_gear.cad_import.export_model_step',side_effect=lambda model,path:Path(path).write_bytes(b'mock_step_only')))
            stack.enter_context(patch('bevel_gear.generated_ltca.ElasticSystem',side_effect=lambda mesh,*a,**k:SimpleNamespace(mesh=mesh,diagnostics=dict(volume_m3=1e-6))))
            generate=stack.enter_context(patch('bevel_gear.generated_ltca.mesh_step_tetra',return_value=mesh))
            prepared=prepare_generated_ltca(requests['pinion'],requests['wheel'],pinion_blank_settings=settings,wheel_blank_settings=settings,
                pinion_material=dict(young_modulus_pa=210e9,poisson_ratio=.3),wheel_material=dict(young_modulus_pa=210e9,poisson_ratio=.3),
                cad_models=models,directory=directory,refinement_regions=regions)
            return prepared,generate.call_args_list

    def test_member_regions_forward_to_the_matching_mesh_without_removing_original_control(self):
        region=dict(refinement_polylines_m=[[[0.,0.,0.],[.001,0.,0.]]],refined_size_m=.000075,
            refinement_core_radius_m=.0003,refinement_radius_m=.0012)
        prepared,calls=self.prepared({'pinion':[region]})
        self.assertEqual(calls[0].kwargs['refinement_regions'],[region])
        self.assertNotIn('refinement_regions',calls[1].kwargs)
        self.assertIn('refinement_points_m',calls[0].kwargs)
        self.assertEqual(calls[0].kwargs['refined_size_m'],.0003)
        self.assertEqual(prepared.diagnostics['additional_refinement_region_counts'],dict(pinion=1,wheel=0))

    def test_none_and_empty_mapping_keep_original_mesh_call_and_payload(self):
        for regions in (None,{},dict(pinion=[],wheel=[])):
            with self.subTest(regions=regions):
                prepared,calls=self.prepared(regions)
                self.assertTrue(all('refinement_regions' not in call.kwargs for call in calls))
                self.assertNotIn('additional_refinement_region_counts',prepared.diagnostics)

    def test_invalid_mapping_and_reused_mesh_cannot_silently_apply_new_regions(self):
        from bevel_gear.generated_ltca import prepare_generated_ltca
        kwargs=dict(pinion_request={},wheel_request={},pinion_blank_settings={},wheel_blank_settings={},pinion_material={},wheel_material={},directory='unused')
        for regions in ([],{'other':[]},{'pinion':None}):
            with self.subTest(regions=regions),self.assertRaisesRegex(ValueError,'pinion/wheel'):
                prepare_generated_ltca(**kwargs,refinement_regions=regions)
        with self.assertRaisesRegex(ValueError,'复用体网格不能'):
            prepare_generated_ltca(**kwargs,meshes={},refinement_regions={'pinion':[{}]})


if __name__=='__main__':unittest.main()
