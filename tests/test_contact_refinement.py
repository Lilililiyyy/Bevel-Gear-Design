"""窄带控制的采样连续性、整轮坐标和真实点核验；不求压力/柔度。"""
from dataclasses import dataclass
import importlib.util
from types import SimpleNamespace
import unittest
from unittest.mock import patch


@dataclass
class Request:tooth_count:int
@dataclass
class Blank:bore_diameter_m:float


@unittest.skipUnless(importlib.util.find_spec('numpy'),'需要NumPy')
class ContactRefinementTests(unittest.TestCase):
    def fixture(self,*,corrupt=False,reject=False,empty=False):
        import numpy as np
        class Surface:
            def Bounds(self):return (0.,1.,0.,1.)
            def Value(self,u,v):return SimpleNamespace(Coord=lambda:(4*u,2*v,0.))
        models={m:SimpleNamespace(request=Request(z),settings=Blank(.001)) for m,z in [('pinion',17),('wheel',47)]}
        prepared=SimpleNamespace(tca=SimpleNamespace(cad_models=models),trimmed={m:SimpleNamespace(surface=Surface(),inside=lambda uv,p:not reject) for m in models})
        def rotation(angle):
            c=np.cos(angle);s=np.sin(angle);return np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])
        def frames(tca,angle,phi,*,tooth_offset=0):
            return {**{m+'_rotation_rows':rotation(angle+.3+tooth_offset*.2).tolist() for m in models},'translation_m':[0.,0.,0.]}
        def geometry(surface,angle,torque,**kwargs):
            uv=np.array([[[i/4,j/2] for i in range(5)] for j in range(3)])
            local=np.dstack((uv[:,:,0]*.004,uv[:,:,1]*.002,np.zeros((3,5))))
            valid=np.ones((3,5),dtype=bool);valid[:,2]=False
            if empty:valid[:]=False
            patches=[]
            for k in kwargs['tooth_offsets']:
                f=frames(prepared.tca,angle,0.,tooth_offset=k);R=np.asarray(f['pinion_rotation_rows']);world=local@R.T
                if corrupt:world[0,0]+=.0001
                patches.append(dict(tooth_offset=k,frames=f,valid_mask=valid,first_uv=uv,second_uv=uv,
                    first_points_world_m=world,second_points_world_m=world))
            return dict(reference_follower_angle_rad=0.,patches=patches,grid_alignment=dict(method='actual_gap_hessian_principal_axes'))
        return prepared,geometry,frames

    def test_invalid_samples_split_both_row_and_column_tracks(self):
        import numpy as np
        from bevel_gear.contact_refinement import continuous_valid_tracks
        points=np.array([[[i,j,0.] for i in range(5)] for j in range(3)])
        valid=np.ones((3,5),dtype=bool);valid[:,2]=False
        tracks,segments,proof=continuous_valid_tracks(points,valid)
        row=[segment for segment in segments if segment['axis']=='row' and segment['fixed_index']==0]
        self.assertEqual([(r['start_index'][1],r['end_index'][1]) for r in row],[(0,1),(3,4)])
        self.assertFalse(proof['invalid_sample_bridged']);self.assertEqual(proof['valid_point_count'],12)
        for line in tracks:self.assertTrue(all(p[0]!=2 for p in line))

    def test_actual_nodes_use_whole_gear_frame_even_when_offset_zero_is_not_selected(self):
        from bevel_gear.contact_refinement import create_contact_refinement
        prepared,geometry,frames=self.fixture()
        with patch('bevel_gear.contact_plane.build_principal_aligned_patches',side_effect=geometry),patch('bevel_gear.generated_tca.generated_tca_frames',side_effect=frames):
            result=create_contact_refinement(prepared,[0.,.1],tooth_offsets=[-1,1],half_widths_m=(.002,.001),
                longitudinal_step_m=.001,transverse_step_m=.001)
        document=result['document'];proof=result['diagnostics']
        self.assertEqual(document['source_driver_angles_rad'],[0.,.1]);self.assertEqual(proof['completed_phase_count'],2)
        self.assertEqual(document['manufacturing']['pinion']['request'],dict(tooth_count=17))
        self.assertFalse(proof['pressure_solved']);self.assertFalse(proof['actual_loaded_region_coverage_certified'])
        for phase in proof['phases']:
            for row in phase['patches']:
                for member in row['members'].values():
                    self.assertEqual(member['finite_BRep_classification_checks'],12)
                    self.assertLess(member['maximum_original_surface_point_error_m'],1e-14)
                    self.assertLess(member['body_frame_roundtrip_error_m'],1e-14)
                    self.assertGreater(member['maximum_error_if_patch_frame_were_used_m'],.0001)

    def test_non_surface_nodes_and_false_material_classification_are_rejected(self):
        from bevel_gear.contact_refinement import create_contact_refinement
        for options in (dict(corrupt=True),dict(reject=True)):
            prepared,geometry,frames=self.fixture(**options)
            with patch('bevel_gear.contact_plane.build_principal_aligned_patches',side_effect=geometry),patch('bevel_gear.generated_tca.generated_tca_frames',side_effect=frames):
                with self.assertRaises(ArithmeticError):create_contact_refinement(prepared,[0.])

    def test_phase_with_no_continuous_control_cannot_be_silently_omitted(self):
        from bevel_gear.contact_refinement import create_contact_refinement
        prepared,geometry,frames=self.fixture(empty=True)
        with patch('bevel_gear.contact_plane.build_principal_aligned_patches',side_effect=geometry),patch('bevel_gear.generated_tca.generated_tca_frames',side_effect=frames):
            with self.assertRaisesRegex(ArithmeticError,'不能跳过'):create_contact_refinement(prepared,[0.])

    def test_invalid_phases_offsets_and_lengths_fail_before_geometry(self):
        from bevel_gear.contact_refinement import create_contact_refinement
        prepared,geometry,frames=self.fixture()
        for angles,kwargs in [([.1,0.],{}),([0.,0.],{}),([0.],dict(tooth_offsets=[0,0])),([0.],dict(transverse_step_m=0.))]:
            with self.assertRaises(ValueError):create_contact_refinement(prepared,angles,**kwargs)


if __name__=='__main__':unittest.main()
