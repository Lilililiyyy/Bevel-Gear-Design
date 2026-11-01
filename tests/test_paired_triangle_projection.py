"""逐行批量点面投影必须保持原点、边和顶点最近点分支。"""
import importlib.util
import unittest

@unittest.skipUnless(importlib.util.find_spec('numpy'),'需要NumPy')
class PairedTriangleProjectionTests(unittest.TestCase):
    def test_random_paired_points_match_original_scalar_path(self):
        import numpy as np
        from bevel_gear.contact_mesh import _closest_on_triangles
        rng=np.random.default_rng(1229);count=4096
        triangles=rng.normal(size=(count,3,3))*.001
        points=rng.normal(size=(count,3))*.002
        # 包括恰好顶点、边中点和三角形内部的零距情况。
        points[0::4]=triangles[0::4,0]
        points[1::4]=triangles[1::4,:2].mean(axis=1)
        points[2::4]=triangles[2::4].mean(axis=1)
        batch=_closest_on_triangles(points,triangles,np)
        expected=[_closest_on_triangles(p,t[None,:,:],np) for p,t in zip(points,triangles)]
        for component in range(3):
            reference=np.concatenate([row[component] for row in expected],axis=0)
            np.testing.assert_allclose(batch[component],reference,rtol=2e-14,atol=1e-20)
        np.testing.assert_allclose(batch[1].sum(axis=1),1.,rtol=0,atol=5e-16)
        self.assertGreaterEqual(float(batch[1].min()),-1e-14)

    def test_shared_single_point_and_bad_pair_layout(self):
        import numpy as np
        from bevel_gear.contact_mesh import _closest_on_triangles
        triangles=np.array([[[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]],[[0.,0.,1.],[1.,0.,1.],[0.,1.,1.]]])
        point=np.array([.25,.25,.4])
        shared=_closest_on_triangles(point,triangles,np)
        for alternative in (point[None,:],np.tile(point,(2,1))):
            paired=_closest_on_triangles(alternative,triangles,np)
            for a,b in zip(shared,paired):np.testing.assert_array_equal(a,b)
        with self.assertRaises(ValueError):_closest_on_triangles(np.zeros((3,3)),triangles,np)

if __name__=='__main__':unittest.main()
