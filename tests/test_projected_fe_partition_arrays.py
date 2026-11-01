"""批量FE交片的零K数学回归；原冻结算法为独立逐片参考。"""
import copy
import io
import unittest

import numpy as np
from bevel_gear.projected_fe_partition import build_projected_fe_contact_system,evaluate_partition_remaining_gaps
from bevel_gear.projected_fe_partition_arrays import ArrayProjectedFEIntegrator,evaluate_array_piece_vertices,typed_array_payload
from test_projected_fe_partition import body,PLANE


def requests(nx=3,ny=2):
    x=np.linspace(-.8,.8,nx+1);y=np.linspace(-.7,.7,ny+1)
    return [dict(tooth_offset=0,row=j,column=i,bounds_xy_m=[x[i],y[j],x[i+1],y[j+1]]) for j in range(ny) for i in range(nx)]


class ArrayPartitionTests(unittest.TestCase):
    def compare(self,first,second,cells):
        reference=build_projected_fe_contact_system(first,second,PLANE,cells)
        integrator=ArrayProjectedFEIntegrator(first,second,PLANE);actual=integrator.build_cells(cells)
        self.assertEqual(actual['piece_count'],reference['piece_count'])
        self.assertEqual(actual['complete_cell_count'],reference['complete_cell_count'])
        self.assertEqual(actual['partial_cell_count'],reference['partial_cell_count'])
        for m in reference['matrices']:
            difference=actual['matrices'][m]-reference['matrices'][m]
            self.assertLess(float(np.max(np.abs(difference.data))) if difference.nnz else 0.,2e-13)
        for old,new in zip(reference['records'],actual['records']):
            self.assertEqual((old['tooth_offset'],old['row'],old['column']),(new['tooth_offset'],new['row'],new['column']))
            self.assertAlmostEqual(old['gap_m'],new['gap_m'],places=13)
            np.testing.assert_allclose(old['first_corner_points_world_m'],new['first_corner_points_world_m'],rtol=0,atol=2e-14)
            for m in reference['matrices']:
                self.assertAlmostEqual(old['levers_m'][m],new['levers_m'][m],places=13)
                self.assertAlmostEqual(old['actual_area_m2'][m],new['actual_area_m2'][m],places=13)
        for old,new in zip(reference['cell_statuses'],actual['cell_statuses']):
            self.assertEqual(old['status'],new['status'])
            self.assertAlmostEqual(old['missing_area_plane_m2'],new['missing_area_plane_m2'],places=13)
            self.assertAlmostEqual(old['integration_weight_sum'],new['integration_weight_sum'],places=13)
        return reference,actual,integrator

    def test_affine_nonmatching_triangles_and_multiple_rectangles(self):
        self.compare(body('pinion',3,4,a=.2,b=-.15),body('wheel',5,2,a=.1,b=.03,c=.1,diagonal=1,reverse=True),requests())

    def test_nonplanar_P1_global_ids_arbitrary_u_and_vertex_extrema(self):
        first=body('pinion',3,4);second=body('wheel',5,3,c=.1,diagonal=1,reverse=True)
        first['nodes_m'][:,2]+=.02*first['nodes_m'][:,0]**2+.03*first['nodes_m'][:,1]*first['nodes_m'][:,0]
        second['nodes_m'][:,2]+=.03*second['nodes_m'][:,1]**2-.02*second['nodes_m'][:,0]*second['nodes_m'][:,1]
        for b in (first,second):
            b['global_node_ids']=np.arange(len(b['nodes_m']))*2+7;b['total_node_count']=int(b['global_node_ids'].max())+3
        reference,actual,_=self.compare(first,second,requests(4,3))
        rng=np.random.default_rng(812);u={m:rng.normal(0,2e-6,(w.shape[0]//3,3)) for m,w in reference['matrices'].items()}
        old=evaluate_partition_remaining_gaps(reference,u,1e-7);new=evaluate_array_piece_vertices(actual,u,1e-7,return_vertex_values=True)
        for a,b in zip(old['cell_checks'],new['cell_checks']):
            self.assertAlmostEqual(a['mean_remaining_gap_m'],b['mean_remaining_gap_m'],places=13)
            self.assertAlmostEqual(a['minimum_remaining_gap_m'],b['minimum_remaining_gap_m'],places=13)
            self.assertEqual(a['checked_piece_vertices'],b['checked_piece_vertices'])
        self.assertFalse(new['loaded_finite_rotation_rebuilt']);self.assertFalse(new['whole_flank_nonpenetration_certified'])
        # 每个原piece的每个顶点也逐项对照，不能只有恰巧相同的cell极小值。
        lookup={(int(c),int(actual['piece_facet_indices']['pinion'][i]),int(actual['piece_facet_indices']['wheel'][i])):i for i,c in enumerate(actual['piece_cell_indices'])}
        for record in reference['records']:
            requested=record['coverage']['requested_cell_index']
            for q in record['quadrature']:
                key=(requested,q['mappings']['pinion']['facet_index'],q['mappings']['wheel']['facet_index']);index=lookup[key]
                start,end=actual['piece_vertex_offsets'][index:index+2];xy=actual['piece_vertices_xy_m'][start:end]
                gap=q['piece_vertex_gap_m'].copy()
                for m in u:gap+=q['piece_vertex_shape_weights'][m]@u[m][q['mappings'][m]['nodes']]@q['coefficients_local'][m]
                gap-=1e-7*q['piece_vertex_levers_m']['wheel']
                for p,value in zip(q['piece_vertices_xy_m'],gap):
                    nearest=int(np.argmin(np.linalg.norm(xy-p,axis=1)))
                    self.assertLess(np.linalg.norm(xy[nearest]-p),1e-14)
                    self.assertAlmostEqual(value,new['vertex_remaining_gaps_m'][start+nearest],places=13)

    def test_hole_and_outside_cells_keep_missing_and_no_renormalization(self):
        first=body('pinion',2,2);first['triangles']=np.delete(first['triangles'],[0,1],axis=0)
        second=body('wheel',3,4,c=.1,reverse=True)
        cells=[dict(tooth_offset=0,row=0,column=0,bounds_xy_m=[-1,-1,0,0]),dict(tooth_offset=0,row=0,column=1,bounds_xy_m=[0,0,.8,.8]),dict(tooth_offset=0,row=0,column=2,bounds_xy_m=[2,2,3,3])]
        old,new,_=self.compare(first,second,cells)
        self.assertEqual(new['partial_cell_count'],2)
        self.assertEqual(new['cell_statuses'][0]['integration_weight_sum'],0.)
        self.assertEqual(new['cell_statuses'][2]['integration_weight_sum'],0.)
        self.assertEqual(new['matrices']['pinion'].shape[1],1)
        self.assertFalse(new['partial_cell_renormalization_used'])

    def test_partial_nonzero_pieces_are_preserved_but_not_in_W(self):
        first=body('pinion',2,2);second=body('wheel',3,4,c=.1,reverse=True)
        cells=[dict(tooth_offset=0,row=0,column=0,bounds_xy_m=[.5,.5,1.5,1.5])]
        old,new,_=self.compare(first,second,cells)
        self.assertGreater(new['piece_count'],0);self.assertEqual(new['complete_cell_count'],0)
        self.assertAlmostEqual(new['cell_statuses'][0]['integration_weight_sum'],.25)
        self.assertGreater(len(new['piece_vertices_xy_m']),0)
        self.assertEqual(new['matrices']['pinion'].shape[1],0)

    def test_overlapping_pressure_cells_rejected(self):
        cells=[dict(tooth_offset=0,row=0,column=i,bounds_xy_m=[-.5,-.5,.5,.5]) for i in range(2)]
        with self.assertRaisesRegex(ValueError,'正面积重叠'):
            ArrayProjectedFEIntegrator(body('pinion',1,1),body('wheel',1,1,c=.1,reverse=True),PLANE).build_cells(cells)

    def test_numeric_only_payload_roundtrips_without_pickle(self):
        _,actual,integrator=self.compare(body('pinion',2,3),body('wheel',4,2,c=.1,reverse=True),requests())
        payload=typed_array_payload(actual)
        self.assertTrue(payload);self.assertTrue(all(v.dtype.kind!='O' for v in payload.values()))
        target=io.BytesIO();np.savez_compressed(target,**payload);target.seek(0)
        with np.load(target,allow_pickle=False) as archive:
            for k,value in payload.items():np.testing.assert_array_equal(archive[k],value)
        reused=ArrayProjectedFEIntegrator.from_verified_bodies(integrator.bodies).build_cells(requests())
        np.testing.assert_array_equal(actual['piece_vertices_xy_m'],reused['piece_vertices_xy_m'])
        self.assertFalse(actual['quadrature_object_dictionaries_created']);self.assertFalse(actual['QP_connected'])


if __name__=='__main__':unittest.main(verbosity=2)
