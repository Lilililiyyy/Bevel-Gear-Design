"""附加局部加密与原全轨迹场取Min；空列表不改变原背景场。"""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

READY=all(importlib.util.find_spec(name) for name in ('numpy','scipy','OCP','gmsh'))


@unittest.skipUnless(READY,'需要NumPy/SciPy/OCP/Gmsh')
class MeshRefinementRegionTests(unittest.TestCase):
    def test_native_min_preserves_original_threshold_and_adds_actual_deep_refinement(self):
        import gmsh
        import numpy as np
        from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
        from bevel_gear.cad_import import ImportedCadModel,export_model_step
        from bevel_gear.elasticity import mesh_step_tetra
        with tempfile.TemporaryDirectory() as directory:
            step=Path(directory)/'block.step'
            export_model_step(ImportedCadModel(BRepPrimAPI_MakeBox(3.,3.,3.).Shape(),dict(units='mm')),step)
            original=gmsh.model.mesh.field.setAsBackgroundMesh;seen=[]
            def observe(tag):
                kind=gmsh.model.mesh.field.getType(tag);row=dict(kind=kind)
                if kind=='Min':
                    ids=gmsh.model.mesh.field.getNumbers(tag,'FieldsList')
                    row['thresholds']=[dict(size=gmsh.model.mesh.field.getNumber(int(i),'SizeMin'),
                        core=gmsh.model.mesh.field.getNumber(int(i),'DistMin'),
                        radius=gmsh.model.mesh.field.getNumber(int(i),'DistMax')) for i in ids]
                seen.append(row);return original(tag)
            options=dict(element_size_m=.0015,maximum_nodes=20000,
                refinement_polylines_m=[[[.0003,.0005,.003],[.001,.0005,.003]]],
                refined_size_m=.0004,refinement_core_radius_m=.0006,refinement_radius_m=.0013)
            extra=dict(refinement_polylines_m=[[[.0018,.0025,.003],[.0027,.0025,.003]]],
                refined_size_m=.0002,refinement_core_radius_m=.0006,refinement_radius_m=.0013)
            with patch.object(gmsh.model.mesh.field,'setAsBackgroundMesh',side_effect=observe):
                coarse=mesh_step_tetra(step,**options)
                fine=mesh_step_tetra(step,**options,refinement_regions=[extra])
            self.assertEqual([r['kind'] for r in seen],['Threshold','Threshold','Min'])
            self.assertEqual(seen[-1]['thresholds'],[dict(size=.0004,core=.0006,radius=.0013),dict(size=.0002,core=.0006,radius=.0013)])
            def local_depth_edges(mesh):
                vertices=np.asarray(mesh.nodes_m)[np.asarray(mesh.tetrahedra)]
                basis=np.swapaxes(vertices[:,1:]-vertices[:,:1],1,2)
                lengths=np.linalg.norm(vertices[:,:,None,:]-vertices[:,None,:,:],axis=-1).max(axis=(1,2));values=[]
                for x in (.002,.0022,.0024):
                    point=np.array([x,.0025,.00265]);b=np.linalg.solve(basis,(point-vertices[:,0])[:,:,None])[:,:,0]
                    bary=np.column_stack((1-b.sum(axis=1),b));inside=(bary>=-1e-8).all(axis=1)&(bary<=1+1e-8).all(axis=1)
                    self.assertTrue(inside.any());values.extend(lengths[inside])
                return float(np.median(values))
            self.assertLess(local_depth_edges(fine),.8*local_depth_edges(coarse))
            self.assertNotIn('refinement_regions',coarse.metadata)
            self.assertEqual(fine.metadata['refinement_regions'][0]['refined_size_m'],.0002)

    def test_bad_regions_fail_before_step_parser(self):
        from bevel_gear.elasticity import mesh_step_tetra
        good=dict(refinement_polylines_m=[[[0.,0.,0.],[.001,0.,0.]]],refined_size_m=.0001,
            refinement_core_radius_m=.0003,refinement_radius_m=.001)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'unused.step';path.write_text('先验证输入',encoding='utf-8')
            wrong=[dict(good,refined_size_m=.002),dict(good,refinement_core_radius_m=.00001),
                dict(good,refinement_radius_m=.0003),dict(good,refinement_polylines_m=[]),dict(good,unknown=True)]
            for region in wrong:
                with self.subTest(region=region),self.assertRaises(ValueError):
                    mesh_step_tetra(path,element_size_m=.002,refinement_regions=[region])


if __name__=='__main__':unittest.main()
