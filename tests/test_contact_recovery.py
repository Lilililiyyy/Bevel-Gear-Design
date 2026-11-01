"""工程压力显示的守恒、相位插值和四图顺序验证，不代替接触求解验证。"""
import unittest
import numpy as np
from shapely.geometry import box

from bevel_gear.contact_recovery import (recover_grid_pressure,grid_triangle_field,
    raster_grid,raw_raster,advected_cycle_maximum)
from bevel_gear.contact_display import prepare_pressure_comparison


def moving_field(center,angle):
    """已知刚性平移的圆形压力帽，用于验证轨迹插值方向及幅值。"""
    x=np.linspace(-4.,4.,81);y=np.linspace(-2.,2.,41);xx,yy=np.meshgrid(x,y)
    p=np.sqrt(np.maximum(0.,1.-((xx-center)**2+yy**2)))
    source=dict(valid_mask=np.ones_like(p,bool),points_width_height_m=np.stack([xx,yy],axis=-1),
        cell_area_m2=.01,driver_angle_rad=angle,phase_index=int(angle+1),
        tooth_offset=0,source_state_sha256='synthetic-translation')
    return grid_triangle_field(source,p)


class PressureRecoveryTests(unittest.TestCase):
    def test_force_nonnegative_mask_and_source_unchanged(self):
        p=np.zeros((21,31));p[8:13:2,8:23:2]=1.8e9
        valid=np.ones_like(p,bool);valid[4:6,4:6]=False
        original=p.copy();q,proof=recover_grid_pressure(p,valid,2e-9,(100e-6,20e-6),50e-6)
        np.testing.assert_array_equal(p,original)
        self.assertTrue(np.all(q>=0));self.assertTrue(np.all(q[~valid]==0))
        self.assertAlmostEqual(q.sum()/p.sum(),1.,places=14)
        self.assertLess(abs(proof['relative_force_error']),1e-14)
        self.assertLess(q.max(),p.max());self.assertFalse(proof['recovered_field_is_new_contact_solution'])

    def test_recovered_center_triangles_preserve_reference_integral(self):
        y,x=np.indices((21,31));xy=np.stack([x*100e-6,y*20e-6],axis=-1)
        p=np.zeros((21,31));p[9:12,12:18]=1e9
        valid=np.ones_like(p,bool);q,_=recover_grid_pressure(p,valid,2e-9,(100e-6,20e-6),50e-6)
        source=dict(valid_mask=valid,points_width_height_m=xy,cell_area_m2=2e-9,
            driver_angle_rad=0.,phase_index=0,tooth_offset=0,source_state_sha256='synthetic')
        field=grid_triangle_field(source,q)
        self.assertAlmostEqual(field['reconstructed_reference_integral_n'],p.sum()*2e-9,places=10)

    def test_invalid_and_zero_pressure(self):
        p=np.zeros((5,5));valid=np.ones_like(p,bool)
        q,proof=recover_grid_pressure(p,valid,1.,(1.,1.),1.)
        self.assertEqual(q.sum(),0.);self.assertEqual(proof['recovered_force_n'],0.)
        p[1,1]=-1.
        with self.assertRaises(ValueError):recover_grid_pressure(p,valid,1.,(1.,1.),1.)

    def test_transport_fills_known_moving_patch_without_peak_overshoot(self):
        fields=[moving_field(-2.,-1.),moving_field(2.,1.)]
        grid=raster_grid(box(-4.,-2.,4.,2.),(81,241))
        raw=raw_raster({'source_fields':fields},grid)
        envelope,proof=advected_cycle_maximum(fields,[-1.,1.],grid,subdivisions=16)
        self.assertEqual(raw[40,120],0.)
        self.assertGreater(envelope[40,120],.99)
        self.assertLessEqual(envelope.max(),1.+1e-12)
        self.assertTrue(np.all(envelope>=raw))
        self.assertEqual(proof['display_phase_count'],17);self.assertEqual(proof['new_FE_solves'],0)

    def test_does_not_bridge_missing_or_unloaded_phase(self):
        fields=[moving_field(-2.,-1.),moving_field(2.,1.)]
        grid=raster_grid(box(-4.,-2.,4.,2.),(81,241))
        envelope,proof=advected_cycle_maximum(fields,[-1.,0.,1.],grid,subdivisions=16)
        self.assertEqual(envelope[40,120],0.);self.assertEqual(proof['connected_same_tooth_segments'],0)

    def test_panel_order_and_independent_limits(self):
        outline=box(0.,0.,1.,1.);rows=[]
        for method,multiplier in [('full_gear_fe',3),('local_halfspace',1)]:
            row={'method':method}
            for key,kind,number in [('instantaneous','instantaneous',0),('cycle_maximum','sampled_cycle_maximum',1)]:
                peak=(multiplier+number)*1e8
                field=dict(points_width_height_m=[[0.,0.],[1.,0.],[0.,1.]],triangles=[[0,1,2]],
                    pressure_pa=[0.,peak,0.],driver_angle_rad=0.)
                row[key]=dict(method=method,kind=kind,outline_geometry=outline,source_fields=[field],source_peak_pressure_pa=peak)
            rows.append(row)
        bundle=prepare_pressure_comparison(rows,view='raw',shape=(21,41));panels=bundle['report']['panels']
        self.assertEqual([(p['method'],p['kind']) for p in panels],[
            ('local_halfspace','instantaneous'),('local_halfspace','sampled_cycle_maximum'),
            ('full_gear_fe','instantaneous'),('full_gear_fe','sampled_cycle_maximum')])
        self.assertEqual([p['colour_limit_pa'] for p in panels],[1e8,2e8,3e8,4e8])


if __name__=='__main__':unittest.main()
