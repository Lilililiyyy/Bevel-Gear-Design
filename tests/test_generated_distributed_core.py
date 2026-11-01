"""新分布FE内核的独立低内存方程验证，不依赖运行缓存或任何齿轮软件。"""
from types import SimpleNamespace
import importlib.util
import math
import unittest

_DEPENDENCIES=all(importlib.util.find_spec(name) for name in ('numpy','scipy','vtk'))
if _DEPENDENCIES:
    import numpy as np
    from scipy import sparse
    from bevel_gear.elasticity import ElasticMaterial,ElasticSystem,TetraMesh
    from bevel_gear.halfspace_contact import RectangularContactGrid
    from bevel_gear.generated_distributed_fe import _SurfaceTraceLocator,solve_distributed_contact_system


def _check_native_many_retains_scalar_projection_and_bad_query_indices(test):
    """同native最近面与原投影公式批量执行，不改变逐点权重或隐藏失败。"""
    axis=np.linspace(-.00045,.00045,7)
    nodes=np.array([[.1+x,.2+y,.3+5*x*x+2*y*y] for y in axis for x in axis]);triangles=[]
    for j in range(6):
        for i in range(6):
            a=7*j+i;triangles.extend(((a,a+1,a+8),(a,a+8,a+7)))
    triangles=np.asarray(triangles);vertices=nodes[triangles]
    normals=np.cross(vertices[:,1]-vertices[:,0],vertices[:,2]-vertices[:,0]);normals/=np.linalg.norm(normals,axis=1)[:,None]
    locator=_SurfaceTraceLocator(dict(nodes_m=nodes,triangles=triangles,normals=normals),np.eye(3))
    random=np.random.default_rng(2710);ids=random.integers(0,len(triangles),512)
    bary=random.uniform(-.4,1.4,(len(ids),3));bary[:,0]=1-bary[:,1]-bary[:,2]
    points=np.einsum('ni,nij->nj',bary,vertices[ids]);outward=normals[ids].copy()
    points+=random.uniform(-5e-7,5e-7,(len(ids),1))*outward
    outward[17]*=-1;points[19]+=.01*outward[19]
    many,reasons=locator.locate_many(points,outward,tolerance_m=.001)
    assert [i for i,reason in enumerate(reasons) if reason is not None]==[17,19]
    for i,(point,normal) in enumerate(zip(points,outward)):
        if reasons[i] is not None:
            with test.assertRaises(ArithmeticError):locator.locate(point,normal,tolerance_m=.001)
            assert many[i] is None
            continue
        single=locator.locate(point,normal,tolerance_m=.001)
        assert np.array_equal(many[i]['nodes'],single['nodes'])
        for key in ('weights','mapped_world','distance_m','face_edge_lengths_m'):
            np.testing.assert_allclose(many[i][key],single[key],rtol=0.,atol=2e-14)


def _one_cell_problem():
    """真实弹性四面体；单位一米力臂使广义力矩数值等价法向合力。"""
    length=.001
    mesh=TetraMesh(nodes_m=[[0.,0.,0.],[length,0.,0.],[0.,length,0.],[0.,0.,-length]],
        tetrahedra=[[0,2,1,3]],boundary_triangles=[],boundary_tags=[],metadata={'units':'m'})
    body=ElasticSystem(mesh,ElasticMaterial(210e9,.3),[1,2,3])
    average=np.array([.6,.2,.2]);matrix=sparse.csc_matrix((-average,([2,5,8],[0,0,0])),shape=(12,1))
    quadrature=[];q=1/math.sqrt(3)
    for a,b in ((-q,-q),(q,-q),(q,q),(-q,q)):
        x,y=(.2+.1*a)*length,(.2+.1*b)*length
        mapping=dict(nodes=np.array([0,1,2]),weights=np.array([1-(x+y)/length,x/length,y/length]),
            mapped_world=np.array([x,y,0.]),distance_m=0.,face_edge_lengths_m=np.array([length,length,math.sqrt(2)*length]))
        quadrature.append(dict(gap_m=0.,mappings={m:mapping for m in ('pinion','wheel')},
            coefficients_local={m:np.array([0.,0.,-1.]) for m in ('pinion','wheel')},levers_m={'pinion':1.,'wheel':1.}))
    area=(.2*length)**2
    record=dict(tooth_offset=0,row=0,column=0,gap_m=0.,area_plane_m2=area,
        actual_area_m2={'pinion':area,'wheel':area},levers_m={'pinion':1.,'wheel':1.},quadrature=quadrature,
        maximum_mapping_error_m=0.,first_corner_points_world_m=[[.1*length,.1*length,0.],[.3*length,.1*length,0.],[.3*length,.3*length,0.],[.1*length,.3*length,0.]])
    grid=RectangularContactGrid(1,1,.2*length,.2*length,origin_xy_m=(.2*length,.2*length))
    geometry=dict(patches=[dict(tooth_offset=0,grid=grid,valid_mask=np.ones((1,1),dtype=bool))])
    system=dict(records=[record],matrices={m:matrix for m in ('pinion','wheel')},geometry=geometry,
        maximum_cad_fe_mapping_error_m=0.,pressure_basis='真实FE单面元代数验证')
    prepared=SimpleNamespace(systems={m:body for m in ('pinion','wheel')},
        models={m:SimpleNamespace(request=SimpleNamespace(tooth_count=1)) for m in ('pinion','wheel')})
    return prepared,system,body,area


def _check_p0_pure_fe_schur_work_pressure_and_stiffness(test):
    """独立3×3刚度消元对照，不能以KKT真值冒充其它验证已通过。"""
    prepared,system,body,area=_one_cell_problem();load=10.
    free=body.stiffness[body.free_dofs][:,body.free_dofs].toarray()
    unit=np.array([0.,0.,1.]);expected_C=2*.6**2*float(np.linalg.solve(free,unit)[2])
    result=solve_distributed_contact_system(prepared,system,load,include_body_fields=True)
    assert result['converged'] and result['halfspace_compliance_used'] is False
    np.testing.assert_allclose(result['equilibrium']['forces_n'],[load],rtol=2e-12)
    assert math.isclose(result['elastic_te_rad'],expected_C*load,rel_tol=2e-12)
    assert math.isclose(result['incremental_stiffness_nm_per_rad'],1/expected_C,rel_tol=2e-12)
    assert math.isclose(result['maximum_pressure_pa'],load/area,rel_tol=2e-12)
    assert math.isclose(result['fields'][0]['forces_n'].sum(),load,rel_tol=2e-12)
    assert abs(result['condensed_energy_relative_error'])<1e-12
    assert max(x['relative_free_residual'] for x in result['body_response_checks'].values())<1e-12
    assert result['geometry_linearization_checked'] is False
    assert result['mesh_convergence_verified'] is False
    assert result['external_tooth_coverage_verified'] is False
    assert result['contact_boundary_checks'][0]['touches_window_edge'] is True
    with test.assertRaisesRegex(ValueError,'非零'):solve_distributed_contact_system(prepared,system,0.)


@unittest.skipUnless(_DEPENDENCIES,'需要NumPy/SciPy/VTK')
class TestGeneratedDistributedCore(unittest.TestCase):
    def test_native_many_retains_scalar_projection_and_bad_query_indices(self):
        _check_native_many_retains_scalar_projection_and_bad_query_indices(self)

    def test_p0_pure_fe_schur_work_pressure_and_stiffness(self):
        _check_p0_pure_fe_schur_work_pressure_and_stiffness(self)


if __name__=='__main__':unittest.main()
