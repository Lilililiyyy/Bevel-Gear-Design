"""投影FE分片的小型数学/几何回归；无接触QP、网格生成或体刚度分解。"""
import copy
import unittest

import numpy as np

from bevel_gear.projected_fe_partition import build_projected_fe_contact_system,evaluate_partition_remaining_gaps


PLANE=dict(origin_world_m=[0.,0.,0.],tangent_rows=[[1.,0.,0.],[0.,1.,0.]],normal_world=[0.,0.,1.])


def rotation(axis,angle):
    axis=np.asarray(axis,dtype=float);axis/=np.linalg.norm(axis)
    x,y,z=axis;k=np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
    return np.eye(3)+np.sin(angle)*k+(1-np.cos(angle))*(k@k)


def body(name,nx,ny,*,a=0.,b=0.,c=0.,diagonal=0,extent=1.,reverse=False):
    x,y=np.meshgrid(np.linspace(-extent,extent,nx+1),np.linspace(-extent,extent,ny+1))
    nodes=np.column_stack((x.ravel(),y.ravel(),(a*x+b*y+c).ravel()));triangles=[]
    for j in range(ny):
        for i in range(nx):
            p=j*(nx+1)+i;q=p+1;r=p+nx+1;s=r+1
            triangles.extend([(p,q,s),(p,s,r)] if (i+j+diagonal)%2==0 else [(p,q,r),(q,s,r)])
    triangles=np.asarray(triangles,dtype=int)
    if reverse:triangles=triangles[:,::-1]
    return dict(name=name,nodes_m=nodes,triangles=triangles,rotation_rows=np.eye(3),axis_world=[0.,0.,1.])


def cells(bounds=(-.6,-.5,.7,.8)):
    return [dict(bounds_xy_m=bounds,tooth_offset=0,row=0,column=0)]


class ProjectedFEPartitionTests(unittest.TestCase):
    def build(self,first,second,requested=None,plane=None):
        return build_projected_fe_contact_system(first,second,PLANE if plane is None else plane,cells() if requested is None else requested)

    def test_tilted_affine_gap_area_resultant_and_torque(self):
        first=body('pinion',3,2,a=.2,b=-.1);second=body('wheel',2,3,a=.3,b=.05,c=.002,diagonal=1,reverse=True)
        system=self.build(first,second);r=system['records'][0];xmin,ymin,xmax,ymax=cells()[0]['bounds_xy_m'];area=(xmax-xmin)*(ymax-ymin);x=(xmax+xmin)/2;y=(ymax+ymin)/2
        self.assertEqual(system['partial_cell_count'],0);self.assertGreater(r['coverage']['piece_count'],4)
        self.assertAlmostEqual(r['gap_m'],.1*x+.15*y+.002,places=14)
        self.assertAlmostEqual(sum(q['integration_weight'] for q in r['quadrature']),1.,places=13)
        for member,a,b,sign in (('pinion',.2,-.1,-1.),('wheel',.3,.05,1.)):
            coef=sign*np.array([-a,-b,1.]);loads=system['matrices'][member].toarray()[:,0].reshape(-1,3)
            np.testing.assert_allclose(loads.sum(axis=0),coef,rtol=3e-14,atol=3e-14)
            self.assertAlmostEqual(r['actual_area_m2'][member],area*np.sqrt(1+a*a+b*b),places=13)
            self.assertAlmostEqual(r['levers_m'][member],-float(coef@np.cross([0,0,1],[x,y,0])),places=14)
        self.assertFalse(system['factorization_performed']);self.assertFalse(system['original_NURBS_gap_used'])

    def test_many_different_triangulations_preserve_affine_integrals(self):
        values=[]
        for n,m,d in ((1,1,0),(2,3,1),(5,4,0),(7,2,1)):
            first=body('pinion',n,m,a=.2,b=-.1,diagonal=d)
            second=body('wheel',m,n,a=.3,b=.05,c=.002,diagonal=1-d,reverse=True)
            system=self.build(first,second);r=system['records'][0];work=0.
            # 同一物理仿射位移场在每个真实P1网格上均精确，不能直接比较不同DOF的W。
            for data in (first,second):
                xyz=data['nodes_m'];u=np.column_stack((.1*xyz[:,0]+.2*xyz[:,1],.3*xyz[:,2]-.05,.2*xyz[:,0]-.1*xyz[:,1]))
                work+=float((system['matrices'][data['name']].T@u.ravel()).item())
            values.append([r['gap_m'],r['actual_area_m2']['pinion'],r['actual_area_m2']['wheel'],r['levers_m']['pinion'],r['levers_m']['wheel'],work])
        np.testing.assert_allclose(values,np.broadcast_to(values[0],np.asarray(values).shape),rtol=2e-13,atol=2e-14)

    def test_cross_edge_cell_is_exactly_partitioned(self):
        system=self.build(body('pinion',1,1,a=.1),body('wheel',1,1,b=.2,c=.01,diagonal=1,reverse=True),cells((-.5,-.5,.5,.5)))
        self.assertGreaterEqual(system['piece_count'],4)
        r=system['records'][0];self.assertAlmostEqual(r['gap_m'],.01,places=14)
        self.assertAlmostEqual(r['coverage']['summed_piece_area_plane_m2'],1.,places=14)
        self.assertEqual(r['coverage']['missing_area_plane_m2'],0.)

    def test_hole_cell_is_partial_not_renormalized_or_added_to_W(self):
        first=body('pinion',2,2);second=body('wheel',2,2,c=.1,reverse=True)
        # 删除右上真实面片，投影缺口不能以凸包或假节点补齐。
        keep=[]
        for tri in second['triangles']:
            xy=second['nodes_m'][tri,:2].mean(axis=0)
            if not (xy[0]>0 and xy[1]>0):keep.append(tri)
        second['triangles']=np.asarray(keep)
        requested=cells((-.5,-.5,.5,.5))+[dict(bounds_xy_m=(-.9,-.9,-.6,-.6),row=1,column=0)]
        system=self.build(first,second,requested)
        self.assertEqual(system['complete_cell_count'],1);self.assertEqual(system['partial_cell_count'],1)
        self.assertEqual(system['matrices']['pinion'].shape[1],1)
        partial=system['partial_records'][0]
        self.assertAlmostEqual(partial['coverage']['covered_area_plane_m2'],.75,places=14)
        self.assertAlmostEqual(partial['coverage']['missing_area_plane_m2'],.25,places=14)
        self.assertAlmostEqual(partial['coverage']['integration_weight_sum'],.75,places=14)
        self.assertAlmostEqual(partial['gap_m'],.075,places=14)
        self.assertFalse(system['partial_cell_renormalization_used'])

    def test_outside_cell_has_no_fabricated_zero_gap_record(self):
        system=self.build(body('pinion',1,1),body('wheel',1,1,c=.01,reverse=True),cells((2.,2.,3.,3.)))
        self.assertEqual(system['records'],[]);self.assertEqual(system['matrices']['wheel'].shape[1],0)
        self.assertEqual(system['cell_statuses'][0]['covered_area_plane_m2'],0.)
        self.assertEqual(system['cell_statuses'][0]['missing_area_plane_m2'],1.)

    def test_projected_overlap_is_rejected(self):
        first=body('pinion',1,1);first['triangles']=np.vstack((first['triangles'],first['triangles'][0]))
        with self.assertRaisesRegex(ValueError,'重叠'):self.build(first,body('wheel',1,1,c=.01,reverse=True))

    def test_pressure_cell_overlap_and_fractional_indices_are_rejected(self):
        first=body('pinion',1,1);second=body('wheel',1,1,c=.01,reverse=True)
        requested=cells()+[dict(bounds_xy_m=(-.3,-.3,.3,.3),column=1)]
        with self.assertRaisesRegex(ValueError,'cell存在正面积重叠'):self.build(first,second,requested)
        requested=cells();requested[0]['row']=.5
        with self.assertRaisesRegex(ValueError,'唯一整数'):self.build(first,second,requested)

    def test_vertical_or_folded_faces_are_rejected(self):
        first=body('pinion',1,1);first['nodes_m'][:,[0,2]]=first['nodes_m'][:,[2,0]]
        with self.assertRaisesRegex(ValueError,'退化'):self.build(first,body('wheel',1,1,c=.01,reverse=True))
        first=body('pinion',1,1);first['triangles'][0]=first['triangles'][0,::-1]
        with self.assertRaisesRegex(ValueError,'折叠'):self.build(first,body('wheel',1,1,c=.01,reverse=True))

    def test_same_projected_edge_cannot_have_two_heights(self):
        first=body('pinion',1,1)
        first['nodes_m']=np.vstack((first['nodes_m'][first['triangles'][0]],first['nodes_m'][first['triangles'][1]]+np.array([0,0,.1])))
        first['triangles']=np.array([[0,1,2],[3,4,5]])
        with self.assertRaisesRegex(ValueError,'不同FE高度'):self.build(first,body('wheel',1,1,c=.2,reverse=True))

    def test_gap_is_actual_FE_linear_height_not_underlying_quadratic(self):
        first=body('pinion',1,1);second=body('wheel',1,1,c=0.,reverse=True)
        x,y=second['nodes_m'][:,:2].T;second['nodes_m'][:,2]=1+x*x+y*y
        system=self.build(first,second,cells((-1.,-1.,1.,1.)))
        self.assertAlmostEqual(system['records'][0]['gap_m'],3.,places=14)
        self.assertGreater(abs(system['records'][0]['gap_m']-(1+2/3)),1.)

    def test_global_node_ids_and_local_rotation_preserve_nodal_torque(self):
        first=body('pinion',2,2,a=.2,b=-.1,extent=2.);second=body('wheel',3,2,a=.3,b=.05,c=.1,extent=2.,reverse=True)
        for data,angle in ((first,.19),(second,-.23)):
            data['rotation_rows']=rotation([0,0,1],angle);data['global_node_ids']=2*np.arange(len(data['nodes_m']))+3
            data['total_node_count']=int(data['global_node_ids'].max())+2
            data['axis_world']=np.array([.2,.4,.8])/np.linalg.norm([.2,.4,.8]);data['axis_origin_world_m']=[.03,-.02,.01]
        system=self.build(first,second);r=system['records'][0]
        for data in (first,second):
            name=data['name'];loads=system['matrices'][name].toarray()[:,0].reshape(-1,3)
            actual=loads[data['global_node_ids']]@data['rotation_rows'].T
            xyz=data['nodes_m']@data['rotation_rows'].T
            torque=float(np.sum(np.cross(xyz-np.array(data['axis_origin_world_m']),actual)@data['axis_world']))
            self.assertAlmostEqual(torque,-r['levers_m'][name],places=13)
            mask=np.ones(len(loads),dtype=bool);mask[data['global_node_ids']]=False
            np.testing.assert_array_equal(loads[mask],0.)

    def test_rigid_rotation_finite_difference_matches_gap_lever(self):
        first=body('pinion',3,2,a=.2,b=-.1,extent=2.);second=body('wheel',2,3,a=.3,b=.05,c=.1,reverse=True,extent=2.)
        axis=np.array([.2,.4,.8]);axis/=np.linalg.norm(axis);second['axis_world']=axis;second['axis_origin_world_m']=[.03,-.02,.01]
        base=self.build(first,second);epsilon=2e-6;averages=[]
        for step in (-epsilon,epsilon):
            changed=copy.deepcopy(second);q=rotation(axis,step);o=np.asarray(changed['axis_origin_world_m'])
            changed['rotation_rows']=q@changed['rotation_rows'];changed['translation_world_m']=o-q@o
            averages.append(self.build(first,changed)['records'][0]['gap_m'])
        derivative=(averages[1]-averages[0])/(2*epsilon)
        self.assertAlmostEqual(derivative,-base['records'][0]['levers_m']['wheel'],delta=2e-10)

    def test_arbitrary_nodal_virtual_work_matches_geometry_finite_difference(self):
        first=body('pinion',3,2,a=.2,b=-.1,extent=2.);second=body('wheel',2,3,a=.3,b=.05,c=.1,reverse=True,extent=2.)
        first['rotation_rows']=rotation([0,0,1],.19);second['rotation_rows']=rotation([0,0,1],-.23)
        rng=np.random.default_rng(155);displacements={d['name']:rng.normal(size=d['nodes_m'].shape)*.01 for d in (first,second)}
        base=self.build(first,second);expected=sum(float((base['matrices'][d['name']].T@displacements[d['name']].ravel()).item()) for d in (first,second))
        epsilon=1e-5;averages=[]
        for step in (-epsilon,epsilon):
            changed=[copy.deepcopy(d) for d in (first,second)]
            for data in changed:data['nodes_m']+=step*displacements[data['name']]
            averages.append(self.build(*changed)['records'][0]['gap_m'])
        derivative=(averages[1]-averages[0])/(2*epsilon)
        self.assertAlmostEqual(derivative,expected,delta=2e-10)

    def test_micrometre_scaling_and_translation_preserve_integration(self):
        first=body('pinion',3,2,a=.2,b=-.1);second=body('wheel',2,3,a=.3,b=.05,c=.1,reverse=True)
        base=self.build(first,second);scale=1e-6;shift=np.array([2e-6,-3e-6,4e-6])
        small=[copy.deepcopy(first),copy.deepcopy(second)]
        for data in small:
            data['nodes_m']*=scale;data['translation_world_m']=shift;data['axis_origin_world_m']=shift
        plane=copy.deepcopy(PLANE);plane['origin_world_m']=shift
        requested=cells(tuple(np.array(cells()[0]['bounds_xy_m'])*scale));actual=self.build(*small,requested,plane)
        r=actual['records'][0];old=base['records'][0]
        self.assertAlmostEqual(r['gap_m']/scale,old['gap_m'],places=12)
        for name in ('pinion','wheel'):
            np.testing.assert_allclose(actual['matrices'][name].toarray(),base['matrices'][name].toarray(),rtol=2e-12,atol=2e-13)
            self.assertAlmostEqual(r['actual_area_m2'][name]/scale**2,old['actual_area_m2'][name],places=12)
            self.assertAlmostEqual(r['levers_m'][name]/scale,old['levers_m'][name],places=12)

    def test_vertex_gap_check_detects_penetration_hidden_by_positive_mean(self):
        first=body('pinion',2,2);second=body('wheel',2,3,a=1.,c=.1,reverse=True)
        system=self.build(first,second,cells((-.5,-.5,.5,.5)))
        result=evaluate_partition_remaining_gaps(system,{d['name']:np.zeros_like(d['nodes_m']) for d in (first,second)},0.)
        self.assertAlmostEqual(result['cell_checks'][0]['mean_remaining_gap_m'],.1,places=14)
        self.assertAlmostEqual(result['minimum_remaining_gap_m'],-.4,places=14)
        self.assertTrue(result['fixed_piecewise_affine_extrema_evaluated'])
        self.assertFalse(result['whole_flank_nonpenetration_certified'])

    def test_vertex_audit_cannot_silently_skip_missing_data(self):
        first=body('pinion',1,1);second=body('wheel',1,1,c=.1,reverse=True)
        system=build_projected_fe_contact_system(first,second,PLANE,cells(),include_vertex_validation=False)
        with self.assertRaisesRegex(ValueError,'未保留'):
            evaluate_partition_remaining_gaps(system,{d['name']:np.zeros_like(d['nodes_m']) for d in (first,second)},0.)


if __name__=='__main__':unittest.main(verbosity=2)
