"""原FE面元压力到连续显示的守恒检查；不把插值当作接触求解。"""
import copy
import unittest

try:
    import numpy as np
except ImportError:
    np=None

from bevel_gear.pressure_field import reconstruct_distributed_fe_pressure_field


@unittest.skipIf(np is None,'本检查需要numpy')
class DistributedPressureDisplayTests(unittest.TestCase):
    def result(self):
        dx=2e-4;dy=1e-4
        y,x=np.meshgrid(np.arange(7)*dy,np.arange(7)*dx,indexing='ij')
        points=np.stack((x,y,.02+x*x),axis=-1)
        p=np.zeros((7,7));p[2:5,2:5]=np.array([[1.,3.,2.],[3.,9.,4.],[1.,2.,1.]])*1e8
        valid=np.ones_like(p,dtype=bool)
        return dict(converged=True,contact_representation='distributed_P0_surface_cells',
            halfspace_compliance_used=False,selected_tooth_offsets=[0],
            fields=[dict(pressure_pa=p,forces_n=p*dx*dy,valid_mask=valid,cell_area_m2=dx*dy)],
            geometry=dict(patches=[dict(tooth_offset=0,first_points_world_m=points,valid_mask=valid)]))

    def test_integral_peak_and_original_values_are_preserved(self):
        source=self.result();before=copy.deepcopy(source)
        field=reconstruct_distributed_fe_pressure_field(source)
        d=field['diagnostics']
        self.assertAlmostEqual(d['original_normal_force_n'],52.)
        self.assertAlmostEqual(d['reconstructed_scalar_integral_n'],52.)
        self.assertTrue(d['all_loaded_center_weights_conserved'])
        self.assertEqual(field['pressure_pa'].max(),9e8)
        ids=field['grid_node_indices'];p=source['fields'][0]['pressure_pa']
        np.testing.assert_array_equal(field['pressure_pa'],p[ids[:,0],ids[:,1]])
        self.assertGreater(d['display_positive_support_area_m2'],d['original_contact_area_m2'])
        for name in ('pressure_pa','forces_n','valid_mask'):
            np.testing.assert_array_equal(source['fields'][0][name],before['fields'][0][name])
        np.testing.assert_array_equal(source['geometry']['patches'][0]['first_points_world_m'],
            before['geometry']['patches'][0]['first_points_world_m'])

    def test_loaded_domain_edge_or_hole_cannot_be_extended_for_display(self):
        for kind in ('edge','hole'):
            with self.subTest(kind=kind):
                source=self.result();field=source['fields'][0]
                if kind=='edge':
                    field['pressure_pa'][0,3]=1e8
                    field['forces_n'][0,3]=1e8*field['cell_area_m2']
                else:
                    field['valid_mask'][1,2]=False
                with self.assertRaisesRegex(ValueError,'邻域|权重'):
                    reconstruct_distributed_fe_pressure_field(source)

    def test_wrong_force_or_solver_identity_is_rejected(self):
        for kind in ('force','node','halfspace'):
            with self.subTest(kind=kind):
                source=self.result()
                if kind=='force':source['fields'][0]['forces_n'][3,3]+=1.
                elif kind=='node':source['contact_representation']='nodal_dual_cells'
                else:source['halfspace_compliance_used']=True
                with self.assertRaises(ValueError):
                    reconstruct_distributed_fe_pressure_field(source)


if __name__=='__main__':unittest.main()
