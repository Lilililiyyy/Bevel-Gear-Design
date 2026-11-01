"""用专业网格器的真实实体检查细化厚度，避免只有表面尺寸变细。"""
import importlib.util
from pathlib import Path
import tempfile
import unittest

AVAILABLE=all(importlib.util.find_spec(name) for name in ('numpy','scipy','OCP','gmsh'))


@unittest.skipUnless(AVAILABLE,'实体网格验证需要NumPy/SciPy/OCP/Gmsh')
class MeshRefinementDepthTests(unittest.TestCase):
    def test_actual_near_field_elements_refine_below_surface(self):
        import numpy as np
        from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
        from bevel_gear.cad_import import ImportedCadModel,export_model_step
        from bevel_gear.elasticity import mesh_step_tetra
        with tempfile.TemporaryDirectory() as directory:
            step=Path(directory)/'block.step'
            export_model_step(ImportedCadModel(BRepPrimAPI_MakeBox(4.,4.,4.).Shape(),dict(units='mm')),step)
            meshes=[]
            for core in (None,.00075):
                meshes.append(mesh_step_tetra(step,element_size_m=.002,maximum_nodes=10000,
                    refinement_points_m=[[.002,.002,.004]],refined_size_m=.00025,
                    refinement_radius_m=.0015,refinement_core_radius_m=core))
            median_edges=[]
            for mesh in meshes:
                vertices=np.asarray(mesh.nodes_m)[np.asarray(mesh.tetrahedra)]
                basis=np.swapaxes(vertices[:,1:]-vertices[:,0,None,:],1,2)
                lengths=np.max(np.linalg.norm(vertices[:,:,None,:]-vertices[:,None,:,:],axis=-1),axis=(1,2))
                volume=np.abs(np.linalg.det(basis)).sum()/6
                self.assertAlmostEqual(volume/64e-9,1.,delta=1e-10)
                observations=[]
                for depth in (.0005,.0006,.0007):
                    for offset in (-.0001,0.,.0001):
                        point=np.array([.002+offset,.002,.004-depth])
                        weights=np.linalg.solve(basis,(point-vertices[:,0])[:,:,None])[:,:,0]
                        bary=np.column_stack((1-weights.sum(axis=1),weights))
                        contained=np.all(bary>=-1e-8,axis=1)&np.all(bary<=1+1e-8,axis=1)
                        self.assertTrue(contained.any())
                        observations.append(float(lengths[contained].mean()))
                median_edges.append(float(np.median(observations)))
            self.assertLess(median_edges[1],.65*median_edges[0])
            self.assertEqual(meshes[0].metadata['refinement_core_radius_m'],.00025)
            self.assertEqual(meshes[1].metadata['refinement_core_radius_m'],.00075)

    def test_core_radius_must_fit_actual_refinement_region(self):
        from bevel_gear.elasticity import mesh_step_tetra
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'unused.step';path.write_text('输入校验先于STEP解析',encoding='utf-8')
            for core in (0.,-.001,.0001,.0015,float('nan'),True):
                with self.subTest(core=core),self.assertRaises(ValueError):
                    mesh_step_tetra(path,element_size_m=.002,refinement_points_m=[[0,0,0]],
                        refined_size_m=.00025,refinement_radius_m=.0015,refinement_core_radius_m=core)
            with self.assertRaises(ValueError):
                mesh_step_tetra(path,element_size_m=.002,refinement_core_radius_m=.00075)


if __name__=='__main__':unittest.main()
