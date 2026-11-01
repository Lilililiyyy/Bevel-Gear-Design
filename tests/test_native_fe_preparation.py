"""独立两四面体控制：真实单邻接边界、朝外法向及错误网格拒绝。"""
from types import SimpleNamespace
import unittest
import numpy as np
from bevel_gear.native_fe_preparation import verified_tetra_boundary


def fixture():
    return SimpleNamespace(nodes_m=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.],[0.,0.,-1.]]),
        tetrahedra=np.array([[0,1,2,3],[0,2,1,4]]),
        boundary_triangles=np.array([[3,1,0],[0,2,3],[1,2,3],[0,1,4],[4,2,0],[1,2,4]]),
        boundary_tags=np.array([11,12,13,14,15,16]))


class NativeBoundaryTests(unittest.TestCase):
    def test_actual_adjacency_orientation_and_tag_identity(self):
        mesh = fixture(); result = verified_tetra_boundary(mesh,chunk_size=1)
        self.assertEqual(len(result['triangles']),6)
        self.assertAlmostEqual(float(result['areas_m2'].sum()),2.+3.**.5)
        originals = {tuple(sorted(t)): tag for t,tag in zip(mesh.boundary_triangles,mesh.boundary_tags)}
        for tri,n,tag in zip(result['triangles'],result['normals'],result['face_tags']):
            self.assertEqual(tag,originals[tuple(sorted(tri))])
            owner = next(t for t in mesh.tetrahedra if set(tri).issubset(t))
            self.assertGreater(float(n @ (mesh.nodes_m[tri].mean(axis=0)-mesh.nodes_m[owner].mean(axis=0))),0.)
        self.assertTrue(result['metadata']['source_Gmsh_boundary_completely_verified'])

    def test_missing_face_and_internal_face_rejected(self):
        for mode in ('missing','internal'):
            mesh=fixture()
            if mode == 'missing':
                mesh.boundary_triangles=mesh.boundary_triangles[:-1];mesh.boundary_tags=mesh.boundary_tags[:-1]
            else:
                mesh.boundary_triangles=np.vstack((mesh.boundary_triangles,[0,1,2]));mesh.boundary_tags=np.r_[mesh.boundary_tags,17]
            with self.subTest(mode=mode),self.assertRaisesRegex(ValueError,'单邻接'):
                verified_tetra_boundary(mesh)

    def test_duplicate_face_rejected(self):
        mesh=fixture();mesh.boundary_triangles=np.vstack((mesh.boundary_triangles,mesh.boundary_triangles[0]))
        mesh.boundary_tags=np.r_[mesh.boundary_tags,11]
        with self.assertRaisesRegex(ValueError,'重复'):
            verified_tetra_boundary(mesh)

    def test_nonmanifold_and_degenerate_rejected(self):
        mesh=fixture();mesh.nodes_m=np.vstack((mesh.nodes_m,[0.,0.,2.]))
        mesh.tetrahedra=np.vstack((mesh.tetrahedra,[0,1,2,5]))
        with self.assertRaisesRegex(ValueError,'非流形'):
            verified_tetra_boundary(mesh)
        mesh=fixture();mesh.nodes_m[3]=[.2,.2,0.]
        with self.assertRaisesRegex(ValueError,'退化'):
            verified_tetra_boundary(mesh)

    def test_fractional_indices_and_nonfinite_nodes_rejected(self):
        mesh=fixture();mesh.tetrahedra=mesh.tetrahedra.astype(float);mesh.tetrahedra[0,0]=.5
        with self.assertRaisesRegex(ValueError,'整数'):
            verified_tetra_boundary(mesh)
        mesh=fixture();mesh.nodes_m[0,0]=np.nan
        with self.assertRaisesRegex(ValueError,'有限'):
            verified_tetra_boundary(mesh)


if __name__ == '__main__':
    unittest.main()
