"""批量原有理NURBS与OCP D1、标量有界射线的独立数值对照。"""
import importlib.util
import json
from pathlib import Path
import unittest

NUMPY = importlib.util.find_spec('numpy') is not None
SCIPY = importlib.util.find_spec('scipy') is not None
OCP = importlib.util.find_spec('OCP') is not None
if NUMPY:
    import numpy as np


def rational_surface():
    """构造非单位参数域、多跨、非全1权重的实际OCP有理B样条。"""
    from OCP.Geom import Geom_BSplineSurface
    from OCP.gp import gp_Pnt
    from OCP.TColgp import TColgp_Array2OfPnt
    from OCP.TColStd import TColStd_Array2OfReal, TColStd_Array1OfReal, TColStd_Array1OfInteger
    poles=TColgp_Array2OfPnt(1,6,1,5);weights=TColStd_Array2OfReal(1,6,1,5)
    for i in range(1,7):
        for j in range(1,6):
            poles.SetValue(i,j,gp_Pnt(10*(i-1)+.4*np.sin(j),8*(j-1)+.2*np.cos(i),
                3.+.3*(i-1)*(j-1)+1.1*np.sin(.4*i+.3*j)))
            weights.SetValue(i,j,1.+.2*np.sin(i)+.1*np.cos(j)+.02*i*j)
    def knots(values,multiplicities):
        k=TColStd_Array1OfReal(1,len(values));m=TColStd_Array1OfInteger(1,len(values))
        for i,(value,count) in enumerate(zip(values,multiplicities),1):k.SetValue(i,value);m.SetValue(i,count)
        return k,m
    u,mu=knots([-2.,.5,4.],[4,2,4]);v,mv=knots([-3.,-.5,2.,5.],[3,1,1,3])
    return Geom_BSplineSurface(poles,weights,u,v,mu,mv,3,2,False,False)


def plane_surface():
    """毫米平面，实际原生参数为[-2,3]×[4,9]，便于独立判定域外射线。"""
    from OCP.Geom import Geom_BSplineSurface
    from OCP.gp import gp_Pnt
    from OCP.TColgp import TColgp_Array2OfPnt
    from OCP.TColStd import TColStd_Array1OfReal,TColStd_Array1OfInteger
    poles=TColgp_Array2OfPnt(1,2,1,2)
    for i,x in enumerate((10.,50.),1):
        for j,y in enumerate((-20.,20.),1):poles.SetValue(i,j,gp_Pnt(x,y,3.))
    u=TColStd_Array1OfReal(1,2);v=TColStd_Array1OfReal(1,2)
    mu=TColStd_Array1OfInteger(1,2);mv=TColStd_Array1OfInteger(1,2)
    for i,value in enumerate((-2.,3.),1):u.SetValue(i,value);mu.SetValue(i,2)
    for i,value in enumerate((4.,9.),1):v.SetValue(i,value);mv.SetValue(i,2)
    return Geom_BSplineSurface(poles,u,v,mu,mv,1,1,False,False)


def native_d1(surface,uv):
    """独立直接调用OCP，显式完成毫米→m及归一化UV的链式求导。"""
    from OCP.gp import gp_Pnt,gp_Vec
    a,b,c,d=surface.Bounds();points=[];first=[];second=[]
    for u,v in uv:
        point,du,dv=gp_Pnt(),gp_Vec(),gp_Vec()
        surface.D1(a+float(u)*(b-a),c+float(v)*(d-c),point,du,dv)
        points.append(np.asarray(point.Coord())*.001)
        first.append(np.asarray(du.Coord())*((b-a)*.001))
        second.append(np.asarray(dv.Coord())*((d-c)*.001))
    return tuple(np.asarray(value).reshape(-1,3) for value in (points,first,second))


def rotated_frame():
    angle=.37;tilt=.21;c,s=np.cos(angle),np.sin(angle);a,b=np.cos(tilt),np.sin(tilt)
    rz=np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])
    rx=np.array([[1.,0.,0.],[0.,a,-b],[0.,b,a]])
    rotation=rx@rz
    return rotation,np.array([.004,-.003,.002]),rotation[:,:2].T


@unittest.skipUnless(NUMPY and SCIPY and OCP,'需要可选NumPy、SciPy和OCP')
class NurbsBatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rational=rational_surface();cls.plane=plane_surface()
        from bevel_gear.generated_surface import build_generated_surface_grid
        from bevel_gear.cad import build_generated_shell
        path=Path(__file__).resolve().parents[1]/'examples/spiral_17_47_pinion_manufacturing.json'
        request=json.loads(path.read_text(encoding='utf-8'));request.update(tip_relief_m=30e-6,tip_relief_start=.65)
        cls.actual_shell=build_generated_shell(build_generated_surface_grid(request,width_count=9,profile_count=11))

    def assert_d1_matches(self,surface,uv):
        from bevel_gear.nurbs_batch import NurbsSurfaceBatch
        before=tuple(surface.Bounds());actual=NurbsSurfaceBatch(surface).sample(uv);expected=native_d1(surface,uv)
        for first,second in zip(actual,expected):
            self.assertEqual(first.shape,(len(uv),3))
            np.testing.assert_allclose(first,second,rtol=2e-12,atol=2e-14)
        self.assertEqual(tuple(surface.Bounds()),before)

    def test_rational_positions_and_normalized_derivatives_at_boundaries_and_knots(self):
        self.assertTrue(self.rational.IsURational() or self.rational.IsVRational())
        self.assertEqual(tuple(self.rational.Bounds()),(-2.,4.,-3.,5.))
        # 参数域不为[0,1]，且两方向跨度不同；漏掉链式因子会使导数明显错误。
        u=[0.,1.,1e-12,1.-1e-12,2.5/6,2.5/6-1e-11,2.5/6+1e-11]
        v=[0.,1.,2.5/8,5./8]
        edge=np.array([[x,y] for x in u for y in v]);random=np.random.default_rng(429).uniform(0.,1.,(91,2))
        self.assert_d1_matches(self.rational,np.vstack((edge,random)))

    def test_large_rational_batch_retains_every_position_and_derivative(self):
        uv=np.random.default_rng(523).uniform(0.,1.,(4103,2));uv[[0,-1]]=[[0.,1.],[1.,0.]]
        self.assert_d1_matches(self.rational,uv)

    def test_actual_generated_relief_flank_matches_native_ocp_d1(self):
        uv=np.vstack(([[0.,0.],[0.,1.],[1.,0.],[1.,1.]],np.random.default_rng(731).uniform(0.,1.,(53,2))))
        self.assert_d1_matches(self.actual_shell.surfaces['left_flank'],uv)
        self.assert_d1_matches(self.actual_shell.surfaces['right_flank'],uv)

    def test_empty_and_invalid_normalized_parameters_do_not_extrapolate(self):
        from bevel_gear.nurbs_batch import NurbsSurfaceBatch
        evaluator=NurbsSurfaceBatch(self.rational)
        self.assertTrue(all(value.shape==(0,3) for value in evaluator.sample(np.empty((0,2)))))
        for uv in ([[-1e-10,.5]],[[.5,1.+1e-10]],[[float('nan'),.5]],[[.5,float('inf')]],[[.5,.5,.5]]):
            with self.subTest(uv=uv),self.assertRaises(ValueError):evaluator.sample(uv)

    def test_bounded_plane_rays_match_scalar_for_mixed_inside_and_outside_targets(self):
        from bevel_gear.nurbs_batch import ray_batch
        from bevel_gear.generated_halfspace import _ray
        rotation,origin,basis=rotated_frame();uv=np.vstack(([[0.,0.],[0.,1.],[1.,0.],[1.,1.]],
            np.random.default_rng(894).uniform(.05,.95,(17,2))))
        local=native_d1(self.plane,uv)[0];targets=(local@rotation.T-origin)@basis.T
        targets=np.vstack((targets,targets[4]+[-.1,0.],targets[4]+[.1,0.],targets[4]+[0.,.1]))
        seeds=np.full((len(targets),2),.5);tolerance=1e-12
        batch=ray_batch(self.plane,rotation,origin,basis,targets,seeds,tolerance)
        scalar=[_ray(self.plane,rotation,origin,basis,target,seed,tolerance) for target,seed in zip(targets,seeds)]
        expected=np.array([item['converged'] for item in scalar])
        np.testing.assert_array_equal(batch['converged'],expected)
        self.assertTrue(expected[:-3].all());self.assertFalse(expected[-3:].any())
        self.assertTrue(np.all((batch['uv']>=0.)&(batch['uv']<=1.)))
        for i,item in enumerate(scalar):
            if item['converged']:
                np.testing.assert_allclose(batch['world'][i],item['world'],rtol=0.,atol=2e-12)
                self.assertLessEqual(batch['residual_m'][i],tolerance)
            else:
                self.assertEqual(batch['status'][i],item['status'])
                self.assertGreater(batch['residual_m'][i],tolerance)

    def test_curved_rational_rays_match_scalar_and_known_surface_intersections(self):
        from bevel_gear.nurbs_batch import ray_batch
        from bevel_gear.generated_halfspace import _ray
        rotation,origin,basis=rotated_frame();uv=np.random.default_rng(118).uniform(.1,.9,(31,2))
        local=native_d1(self.rational,uv)[0];targets=(local@rotation.T-origin)@basis.T
        seeds=np.clip(uv+np.random.default_rng(127).uniform(-.13,.13,uv.shape),0.,1.);tolerance=1e-12
        batch=ray_batch(self.rational,rotation,origin,basis,targets,seeds,tolerance)
        self.assertTrue(batch['converged'].all())
        np.testing.assert_allclose(batch['local'],local,rtol=0.,atol=5e-11)
        for i,(target,seed) in enumerate(zip(targets,seeds)):
            scalar=_ray(self.rational,rotation,origin,basis,target,seed,tolerance)
            self.assertTrue(scalar['converged'])
            np.testing.assert_allclose(batch['world'][i],scalar['world'],rtol=0.,atol=5e-12)
            np.testing.assert_allclose(batch['uv'][i],scalar['uv'],rtol=0.,atol=5e-10)

    def test_actual_relief_flank_rays_match_scalar_in_the_local_tangent_chart(self):
        from bevel_gear.nurbs_batch import ray_batch
        from bevel_gear.generated_halfspace import _ray
        surface=self.actual_shell.surfaces['left_flank'];rotation=rotated_frame()[0]
        centre=np.array([.5,.8]);point,du,dv=native_d1(surface,centre[None,:]);origin=point[0]@rotation.T
        first=du[0]@rotation.T;second=dv[0]@rotation.T;normal=np.cross(first,second);normal/=np.linalg.norm(normal)
        first/=np.linalg.norm(first);basis=np.array([first,np.cross(normal,first)])
        uv=centre+np.random.default_rng(245).uniform(-.025,.025,(19,2))
        local=native_d1(surface,uv)[0];targets=(local@rotation.T-origin)@basis.T;seeds=np.tile(centre,(len(uv),1))
        batch=ray_batch(surface,rotation,origin,basis,targets,seeds,1e-12)
        self.assertTrue(batch['converged'].all())
        np.testing.assert_allclose(batch['local'],local,rtol=0.,atol=5e-11)
        for i,target in enumerate(targets):
            scalar=_ray(surface,rotation,origin,basis,target,seeds[i],1e-12)
            self.assertTrue(scalar['converged'])
            np.testing.assert_allclose(batch['world'][i],scalar['world'],rtol=0.,atol=5e-12)

    def test_degenerate_ray_graph_fails_instead_of_accepting_a_clamped_projection(self):
        from bevel_gear.nurbs_batch import ray_batch
        from bevel_gear.generated_halfspace import _ray
        rotation=np.eye(3);origin=np.zeros(3);basis=np.array([[0.,0.,1.],[0.,1.,0.]])
        target=np.array([[.004,0.]]);seed=np.array([[.5,.5]])
        batch=ray_batch(self.plane,rotation,origin,basis,target,seed,1e-12)
        scalar=_ray(self.plane,rotation,origin,basis,target[0],seed[0],1e-12)
        self.assertFalse(batch['converged'][0]);self.assertFalse(scalar['converged'])
        self.assertEqual(batch['status'][0],'ray_graph_degenerate')
        self.assertEqual(batch['status'][0],scalar['status'])


if __name__=='__main__':unittest.main()
