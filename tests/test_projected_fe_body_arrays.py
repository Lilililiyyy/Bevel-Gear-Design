"""全量批量投影图先验回归；冻结逐pair类为独立判断参照，0K。"""
import copy
import unittest
import numpy as np
from bevel_gear.projected_fe_partition import ProjectedFacetBody
from bevel_gear.projected_fe_body_arrays import ArrayProjectedFacetBody
from bevel_gear.projected_fe_partition_arrays import ArrayProjectedFEIntegrator,evaluate_array_piece_vertices
from test_projected_fe_partition import body,PLANE

OPTIONS=dict(minimum_normal_cosine=1e-6,overlap_relative_tolerance=256*np.finfo(float).eps,barycentric_tolerance=5e-11)
P=(np.asarray(PLANE['origin_world_m']),np.asarray(PLANE['tangent_rows']),np.asarray(PLANE['normal_world']))


class BodyArrayTests(unittest.TestCase):
    def test_original_fields_queries_and_complete_P1_operator_equal(self):
        inputs=[body('pinion',6,4,a=.1,b=-.03),body('wheel',5,7,a=.15,b=.02,c=.1,reverse=True)]
        inputs[0]['nodes_m'][:,2]+=.03*inputs[0]['nodes_m'][:,0]**2
        old=[ProjectedFacetBody(b,P,**OPTIONS) for b in inputs];new=[ArrayProjectedFacetBody(b,P,**OPTIONS) for b in inputs]
        for a,b in zip(old,new):
            for name in ('nodes','triangles','rotation','node_ids','world_nodes','xy_nodes','heights','normals','cosines','projected_origins','projected_inverse','face_edge_lengths'):
                np.testing.assert_array_equal(getattr(a,name),getattr(b,name))
            for xy in ((.2,.3),(-.6,.1)):
                x=a.ray(xy);y=b.ray(xy)
                np.testing.assert_array_equal(x['mapped_world'],y['mapped_world'])
            self.assertEqual(a.projection_audit['queried_neighbor_pairs'],b.projection_audit['queried_neighbor_pairs'])
            self.assertTrue(b.projection_audit['all_original_facets_retained'])
        cells=[dict(tooth_offset=0,row=0,column=i,bounds_xy_m=[-.5+i*.5,-.3,i*.5,.3]) for i in range(2)]
        a=ArrayProjectedFEIntegrator.from_verified_bodies(old).build_cells(cells)
        b=ArrayProjectedFEIntegrator.from_verified_bodies(new).build_cells(cells)
        self.assertEqual(a['piece_count'],b['piece_count'])
        for m,w in a['matrices'].items():self.assertEqual((w-b['matrices'][m]).nnz,0)
        np.testing.assert_array_equal(a['piece_vertices_xy_m'],b['piece_vertices_xy_m'])
        rng=np.random.default_rng(9);u={m:rng.normal(0,1e-8,(w.shape[0]//3,3)) for m,w in a['matrices'].items()}
        x=evaluate_array_piece_vertices(a,u,1e-7);y=evaluate_array_piece_vertices(b,u,1e-7)
        self.assertEqual(x,y)

    def assert_both_reject(self,data):
        for cls in (ProjectedFacetBody,ArrayProjectedFacetBody):
            with self.subTest(cls=cls.__name__):
                with self.assertRaises((ValueError,ArithmeticError)):cls(data,P,**OPTIONS)

    def test_positive_projected_overlap_is_rejected(self):
        data=body('pinion',1,1);data['triangles']=np.vstack((data['triangles'],data['triangles'][0]))
        self.assert_both_reject(data)

    def test_shared_projected_edge_height_jump_is_rejected(self):
        data=dict(name='pinion',nodes_m=np.array([[-1,-1,0],[0,-1,0],[0,1,0],[0,-1,.001],[1,-1,.001],[0,1,.001]],dtype=float),
            triangles=np.array([[0,1,2],[3,4,5]],dtype=int))
        self.assert_both_reject(data)

    def test_folded_or_degenerate_or_wrong_normal_is_rejected(self):
        folded=body('pinion',2,2);folded['triangles'][0]=folded['triangles'][0,::-1];self.assert_both_reject(folded)
        degenerate=body('pinion',2,2);degenerate['triangles'][0]=[0,0,1];self.assert_both_reject(degenerate)
        normal=body('pinion',2,2);normal['outward_normals_local']=np.tile([.1,0,np.sqrt(.99)],(len(normal['triangles']),1));self.assert_both_reject(normal)

    def test_bad_global_ids_rotation_and_index_are_rejected(self):
        wrong=body('pinion',2,2);wrong['global_node_ids']=np.zeros(len(wrong['nodes_m']),dtype=int);self.assert_both_reject(wrong)
        wrong=body('pinion',2,2);wrong['rotation_rows']=np.eye(3)*2;self.assert_both_reject(wrong)
        wrong=body('pinion',2,2);wrong['triangles'][0,0]=999;self.assert_both_reject(wrong)


if __name__=='__main__':unittest.main(verbosity=2)
