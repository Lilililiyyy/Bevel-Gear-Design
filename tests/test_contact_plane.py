"""核验投影坐标旋转、真实查询主轴及非椭圆/边缘拒绝分支。"""
import math
import unittest
try:
    import numpy as np
except ImportError:np=None
from bevel_gear.contact_plane import rotate_plane_basis,principal_gap_direction


@unittest.skipIf(np is None,'需要NumPy')
class ContactPlaneTests(unittest.TestCase):
    def test_rotated_basis_preserves_world_point_and_normal(self):
        old=np.array([[1.,0.,0.],[0.,1.,0.]])
        new=rotate_plane_basis(old,-.27);point=np.array([.002,.0003,0.])
        np.testing.assert_allclose((new@point)@new,point,rtol=0,atol=1e-18)
        np.testing.assert_allclose(np.cross(*new),np.cross(*old),rtol=0,atol=1e-15)
        np.testing.assert_array_equal(rotate_plane_basis(old,0.),old)

    def test_nonlinear_gap_selects_direction_without_replacing_gap(self):
        angle=-.23;c=math.cos(angle);s=math.sin(angle);queries=[]
        def actual_gap(x,y):
            queries.append((x,y));u=c*x+s*y;v=-s*x+c*y
            return .5*(2*u*u+100*v*v)+2e5*v**4
        result=principal_gap_direction(actual_gap)
        self.assertAlmostEqual(result['angle_rad'],angle,places=5)
        self.assertEqual(result['actual_gap_queries'],len(queries))
        self.assertGreater(result['actual_gap_queries'],9)
        with self.assertRaises(ArithmeticError):principal_gap_direction(lambda x,y:x*x-y*y)
        with self.assertRaises(ArithmeticError):principal_gap_direction(lambda x,y:x*x+y*y)
        with self.assertRaises(ArithmeticError):principal_gap_direction(lambda x,y:None)


if __name__=='__main__':unittest.main()
