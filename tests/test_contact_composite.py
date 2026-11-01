"""原三角域瞬时/最大压力的数学定义、孔洞、证书范围与独立重绘验证。"""
from copy import deepcopy
import gzip
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

NUMPY = importlib.util.find_spec('numpy') is not None
PLOTS = importlib.util.find_spec('matplotlib') is not None
GEOS = importlib.util.find_spec('shapely') is not None
if NUMPY:
    import numpy as np


def square(values, *, alternate=False, phase=0):
    return dict(points_width_height_m=[[0., 0.], [1., 0.], [1., 1.], [0., 1.]],
        triangles=[[0, 1, 3], [1, 2, 3]] if alternate else [[0, 1, 2], [0, 2, 3]],
        pressure_pa=values, phase_index=phase, tooth_offset=0)


@unittest.skipUnless(NUMPY and PLOTS and GEOS, '需要可选NumPy、Matplotlib和GEOS')
class TrianglePressureCompositeTests(unittest.TestCase):
    def setUp(self):
        import shapely
        from shapely.geometry import box
        self.outline = box(0., 0., 1., 1.)
        self.empty = shapely.GeometryCollection()

    def build(self, fields, *, kind='sampled_cycle_maximum', method='full_gear_fe', unknown=None,
            zero=None, levels=None):
        from bevel_gear.contact_composite import build_triangle_pressure_map
        return build_triangle_pressure_map(fields, method=method, kind=kind,
            outline_geometry=self.outline, levels_pa=np.linspace(0., 100., 17) if levels is None else levels,
            unknown_geometry=self.empty if unknown is None else unknown,
            certified_zero_geometry=zero)

    def test_maximum_superlevel_is_union_not_sum_mean_or_convex_hull(self):
        from bevel_gear.contact_composite import triangle_pressure_superlevel
        from shapely.geometry import Point
        fields=[square([0., 100., 100., 0.]), square([100., 0., 0., 100.], phase=1)]
        area=triangle_pressure_superlevel(fields, 75., outline_geometry=self.outline)
        self.assertAlmostEqual(area.area, .5)
        self.assertTrue(area.covers(Point(.1, .5)))
        self.assertTrue(area.covers(Point(.9, .5)))
        self.assertFalse(area.covers(Point(.5, .5)))
        self.assertEqual(area.geom_type, 'MultiPolygon')

    def test_different_mesh_topologies_match_analytic_maximum_at_random_points(self):
        import shapely
        from bevel_gear.contact_composite import triangle_pressure_superlevel
        first=square([0., 100., 100., 0.])
        second=dict(points_width_height_m=[[0.,0.],[1.,0.],[1.,1.],[0.,1.],[.5,.5]],
            triangles=[[0,1,4],[1,2,4],[2,3,4],[3,0,4]],pressure_pa=[0.,0.,80.,80.,40.])
        points=np.random.default_rng(817).uniform(0.,1.,(1500,2))
        expected=np.maximum(100*points[:,0],80*points[:,1])
        for level in np.linspace(3.,97.,19):
            area=triangle_pressure_superlevel([first,second],float(level),outline_geometry=self.outline)
            np.testing.assert_array_equal(shapely.covers(area,shapely.points(points)),expected>=level)

    def test_single_triangle_threshold_crossing_has_exact_area(self):
        from bevel_gear.contact_composite import triangle_pressure_superlevel
        field=dict(points_width_height_m=[[0.,0.],[1.,0.],[0.,1.]],triangles=[[0,1,2]],pressure_pa=[0.,100.,0.])
        region=triangle_pressure_superlevel([field],50.,outline_geometry=self.outline)
        self.assertAlmostEqual(region.area,.125)

    def test_hole_and_island_are_preserved_at_different_pressure_levels(self):
        from bevel_gear.contact_composite import triangle_pressure_superlevel
        from shapely.geometry import Point,box
        xy=[[0.,0.],[3.,0.],[3.,3.],[0.,3.],[1.,1.],[2.,1.],[2.,2.],[1.,2.]]
        ring=dict(points_width_height_m=xy,triangles=[[0,1,5],[0,5,4],[1,2,6],[1,6,5],
            [2,3,7],[2,7,6],[3,0,4],[3,4,7]],pressure_pa=[100.]*8)
        island=dict(points_width_height_m=[[1.3,1.3],[1.7,1.3],[1.5,1.7]],triangles=[[0,1,2]],pressure_pa=[200.]*3)
        region=triangle_pressure_superlevel([ring,island],50.,outline_geometry=box(0.,0.,3.,3.))
        self.assertAlmostEqual(region.area,8.08)
        self.assertTrue(region.covers(Point(1.5,1.4)))
        self.assertFalse(region.covers(Point(1.1,1.1)))
        high=triangle_pressure_superlevel([ring,island],150.,outline_geometry=box(0.,0.,3.,3.))
        self.assertAlmostEqual(high.area,.08)

    def test_narrow_real_triangle_is_not_deleted_by_display_grid_threshold(self):
        from bevel_gear.contact_composite import triangle_pressure_superlevel
        from shapely.geometry import Point
        field=dict(points_width_height_m=[[.1,.1],[.9,.1],[.5,.100001]],triangles=[[0,1,2]],pressure_pa=[100.]*3)
        region=triangle_pressure_superlevel([field],99.,outline_geometry=self.outline)
        self.assertAlmostEqual(region.area,4e-7,places=16)
        self.assertTrue(region.covers(Point(.5,.1000005)))

    def test_raw_vertex_peak_is_retained_without_growing_a_fictitious_peak_area(self):
        data=self.build([square([100.,0.,0.,0.])])
        self.assertEqual(data['source_peak_pressure_pa'],100.)
        self.assertFalse(data['query_grid_resampling'])
        self.assertFalse(data['original_pressure_values_modified'])
        from bevel_gear.contact_composite import triangle_pressure_superlevel
        top=triangle_pressure_superlevel([square([100.,0.,0.,0.])],100.,outline_geometry=self.outline)
        self.assertEqual(top.area,0.)

    def test_constant_zero_triangles_do_not_become_positive_pressure_support(self):
        data=self.build([square([0.,0.,0.,0.])],zero=self.outline)
        self.assertEqual(data['source_peak_pressure_pa'],0.)
        self.assertTrue(data['positive_support_geometry'].is_empty)
        self.assertTrue(data['certified_zero_geometry'].equals(self.outline))

    def test_original_p0_cell_discontinuity_is_not_averaged_at_duplicate_coordinates(self):
        from bevel_gear.contact_composite import triangle_pressure_superlevel
        from shapely.geometry import box
        field=dict(points_width_height_m=[[0.,0.],[1.,0.],[1.,1.],[0.,1.],
            [1.,0.],[2.,0.],[2.,1.],[1.,1.]],
            triangles=[[0,1,2],[0,2,3],[4,5,6],[4,6,7]],pressure_pa=[100.]*4+[200.]*4)
        before=deepcopy(field)
        low=triangle_pressure_superlevel([field],50.,outline_geometry=box(0.,0.,2.,1.))
        high=triangle_pressure_superlevel([field],150.,outline_geometry=box(0.,0.,2.,1.))
        self.assertEqual(low.area,2.)
        self.assertTrue(high.equals(box(1.,0.,2.,1.)))
        self.assertEqual(field,before)

    def test_overlapping_source_triangles_fail_instead_of_selecting_one_pressure(self):
        from bevel_gear.contact_composite import OriginalP1Contours
        field=dict(points_width_height_m=[[0.,0.],[1.,0.],[0.,1.],[0.,0.],[1.,0.],[0.,1.]],
            triangles=[[0,1,2],[3,4,5]],pressure_pa=[100.]*3+[200.]*3)
        with self.assertRaises(ValueError):OriginalP1Contours(field)

    def test_unknown_domain_defaults_to_unknown_not_zero(self):
        from bevel_gear.contact_composite import build_triangle_pressure_map
        data=build_triangle_pressure_map([square([100.]*4)],method='full_gear_fe',kind='instantaneous',
            outline_geometry=self.outline,levels_pa=[0.,50.,100.])
        self.assertTrue(data['unknown_geometry'].equals(self.outline))
        self.assertTrue(data['certified_zero_geometry'].is_empty)

    def test_instantaneous_domain_uses_only_its_own_explicit_coverage(self):
        from shapely.geometry import box
        instant=self.build([square([100.]*4)],kind='instantaneous',unknown=box(0.,0.,.1,1.))
        cycle=self.build([square([100.]*4),square([100.]*4,phase=1)],unknown=box(0.,0.,.4,1.))
        self.assertAlmostEqual(instant['unknown_projected_area_fraction'],.1)
        self.assertAlmostEqual(cycle['unknown_projected_area_fraction'],.4)
        with self.assertRaises(ValueError):
            self.build([square([100.]*4),square([100.]*4,phase=1)],kind='instantaneous')

    def test_contradictory_zero_or_non_linear_or_clipped_colour_scale_fails(self):
        from shapely.geometry import box
        for zero,unknown in [(self.outline,self.empty),(box(0.,0.,.1,1.),box(0.,0.,.2,1.))]:
            with self.assertRaises(ValueError):self.build([square([100.]*4)],zero=zero,unknown=unknown)
        for levels in ([0.,20.,100.],[0.,50.,90.],[0.,float('nan'),100.],[0.,100.,50.]):
            with self.assertRaises(ValueError):self.build([square([100.]*4)],levels=levels)

    def test_source_arrays_and_clockwise_connectivity_remain_byte_exact(self):
        field=square([0.,100.,100.,0.]);field['triangles']=[[0,2,1],[0,3,2]]
        before=deepcopy(field);data=self.build([field])
        self.assertEqual(field,before)
        np.testing.assert_array_equal(data['source_fields'][0]['triangles'],before['triangles'])
        np.testing.assert_array_equal(data['source_fields'][0]['pressure_pa'],before['pressure_pa'])

    def test_public_si_export_load_and_full_recompute_match(self):
        from bevel_gear.contact_composite import export_triangle_pressure_map,load_triangle_pressure_map
        data=self.build([square([0.,100.,100.,0.]),square([100.,0.,0.,100.],phase=1)])
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'pressure.json';record=export_triangle_pressure_map(data,path)
            self.assertEqual(len(record['sha256']),64)
            restored=load_triangle_pressure_map(path,recompute=True)
            for first,second in zip(data['superlevel_geometries'],restored['superlevel_geometries']):
                self.assertEqual(first.symmetric_difference(second).area,0.)
            content=json.loads(path.read_text(encoding='utf-8'))
            content['source_fields'][0]['pressure_pa'][0]=1.
            path.write_text(json.dumps(content),encoding='utf-8')
            with self.assertRaises(ValueError):load_triangle_pressure_map(path)

    def test_display_only_export_cannot_claim_source_recomputation(self):
        from bevel_gear.contact_composite import export_triangle_pressure_map,load_triangle_pressure_map
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'pressure.json';export_triangle_pressure_map(self.build([square([100.]*4)]),path,include_source_fields=False)
            with self.assertRaises(ValueError):load_triangle_pressure_map(path,recompute=True)

    def test_gzip_decodes_to_identical_si_json_and_recomputes_original_fields(self):
        from bevel_gear.contact_composite import export_triangle_pressure_map,load_triangle_pressure_map
        data=self.build([square([0.,100.,100.,0.]),square([100.,0.,0.,100.],phase=1)])
        with tempfile.TemporaryDirectory() as directory:
            plain=Path(directory)/'pressure.json';packed=Path(directory)/'pressure.json.gz'
            first=export_triangle_pressure_map(data,plain);second=export_triangle_pressure_map(data,packed)
            self.assertEqual(gzip.decompress(packed.read_bytes()),plain.read_bytes())
            self.assertEqual(first['decoded_sha256'],second['decoded_sha256'])
            self.assertEqual(second['compression'],'gzip')
            self.assertEqual(packed.read_bytes()[4:8],b'\0\0\0\0')
            self.assertEqual(packed.read_bytes()[3]&0x18,0)
            restored=load_triangle_pressure_map(packed,recompute=True)
            for a,b in zip(data['source_fields'],restored['source_fields']):
                np.testing.assert_array_equal(a['pressure_pa'],b['pressure_pa'])
                np.testing.assert_array_equal(a['triangles'],b['triangles'])
            for a,b in zip(data['superlevel_geometries'],restored['superlevel_geometries']):
                self.assertEqual(a.symmetric_difference(b).area,0.)
            other=Path(directory)/'another_name.json.gz';export_triangle_pressure_map(data,other)
            self.assertEqual(packed.read_bytes(),other.read_bytes())

    def test_gzip_tamper_crc_non_json_and_wrong_schema_are_rejected(self):
        from bevel_gear.contact_composite import export_triangle_pressure_map,load_triangle_pressure_map
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'pressure.json.gz';export_triangle_pressure_map(self.build([square([100.]*4)]),path)
            original=path.read_bytes();content=json.loads(gzip.decompress(original))
            content['source_fields'][0]['pressure_pa'][0]=99.
            tampered=gzip.compress(json.dumps(content).encode('utf-8'),mtime=0)
            crc=bytearray(original);crc[-8]^=1
            for payload in (tampered,bytes(crc),original[:-5],b'not a gzip stream',
                    gzip.compress(b'not JSON',mtime=0),gzip.compress(b'\x80\x04N.',mtime=0),
                    gzip.compress(b'[]',mtime=0),gzip.compress(b'{"schema":"wrong"}',mtime=0)):
                path.write_bytes(payload)
                with self.subTest(payload_size=len(payload)),self.assertRaises(ValueError):load_triangle_pressure_map(path)

    def test_exported_colour_range_and_band_count_cannot_silently_change(self):
        from bevel_gear.contact_composite import export_triangle_pressure_map,load_triangle_pressure_map
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'pressure.json';export_triangle_pressure_map(self.build([square([100.]*4)]),path)
            original=json.loads(path.read_text(encoding='utf-8'))
            for mutation in ('clipped_range','missing_band','nonlinear'):
                content=deepcopy(original)
                if mutation=='clipped_range':content['source_peak_pressure_pa']=101.
                if mutation=='missing_band':content['superlevel_geometries'].pop()
                if mutation=='nonlinear':content['levels_pa'][1]*=1.2
                path.write_text(json.dumps(content),encoding='utf-8')
                with self.assertRaises(ValueError):load_triangle_pressure_map(path)

    def test_invalid_pressure_coordinates_and_triangle_topology_fail(self):
        from bevel_gear.contact_composite import OriginalP1Contours
        for mutation in ('negative','nan','degenerate','out_of_range'):
            field=square([100.]*4)
            if mutation=='negative':field['pressure_pa'][0]=-1.
            if mutation=='nan':field['points_width_height_m'][0][0]=float('nan')
            if mutation=='degenerate':field['triangles'][0]=[0,0,1]
            if mutation=='out_of_range':field['triangles'][0]=[0,1,99]
            with self.assertRaises(ValueError):OriginalP1Contours(field)

    def test_four_panel_plot_has_one_absolute_scale_and_source_data(self):
        from bevel_gear.contact_composite import plot_contact_composite
        rows=[]
        for method in ('full_gear_fe','local_halfspace'):
            rows.append(dict(method=method,instantaneous=self.build([square([0.,100.,100.,0.])],kind='instantaneous',method=method),
                cycle_maximum=self.build([square([0.,100.,100.,0.]),square([100.,0.,0.,100.],phase=1)],method=method)))
        with tempfile.TemporaryDirectory() as directory:
            record=plot_contact_composite(rows,directory,lang='en')
            self.assertTrue(record['colour_scale_shared_by_all_panels'])
            self.assertAlmostEqual(record['pressure_limit_mpa'],.0001)
            self.assertEqual(len(record['source_pressure_maps']),4)
            self.assertTrue((Path(directory)/'en/17_contact_composite.pdf').is_file())
            bad=deepcopy(rows);bad[1]['instantaneous']['levels_pa']=np.linspace(0.,200.,17)
            with self.assertRaises(ValueError):plot_contact_composite(bad,directory,lang='en')

    def test_four_panel_optional_gzip_exports_four_reproducible_maps(self):
        from bevel_gear.contact_composite import plot_contact_composite,load_triangle_pressure_map
        rows=[]
        for method in ('full_gear_fe','local_halfspace'):
            rows.append(dict(method=method,instantaneous=self.build([square([100.]*4)],kind='instantaneous',method=method),
                cycle_maximum=self.build([square([100.]*4)],method=method)))
        with tempfile.TemporaryDirectory() as directory:
            record=plot_contact_composite(rows,directory,lang='en',source_compression='gzip')
            self.assertEqual(record['source_compression'],'gzip')
            for source in record['source_pressure_maps']:
                self.assertTrue(source['file'].endswith('.json.gz'))
                data=load_triangle_pressure_map(Path(directory)/'data'/source['file'],recompute=True)
                self.assertEqual(data['source_peak_pressure_pa'],100.)

    def test_legend_omits_empty_domains_and_preserves_nonempty_unknown(self):
        from shapely.geometry import box
        from bevel_gear.contact_composite import plot_contact_composite
        # 全面零压和局部未认证两种真实域语义分别检查，压力始终不变。
        for mixed in (False,True):
            rows=[]
            zero=box(0.,0.,.25,1.) if mixed else self.outline
            unknown=box(.25,0.,.5,1.) if mixed else self.empty
            for method in ('full_gear_fe','local_halfspace'):
                rows.append(dict(method=method,
                    instantaneous=self.build([square([0.]*4)],kind='instantaneous',method=method,zero=zero,unknown=unknown),
                    cycle_maximum=self.build([square([0.]*4)],method=method,zero=zero,unknown=unknown)))
            before=deepcopy(rows)
            with tempfile.TemporaryDirectory() as directory:
                record=plot_contact_composite(rows,directory,lang='en')
                self.assertEqual(record['legend_categories'],
                    ['zero_pressure','uncertified','unclassified'] if mixed else ['zero_pressure'])
                svg=(Path(directory)/'en/17_contact_composite.svg').read_text(encoding='utf8')
                self.assertIn('Zero-pressure region',svg)
                self.assertEqual('Uncertified domain' in svg,mixed)
                self.assertEqual('No pressure/zero evidence' in svg,mixed)
            for row,original in zip(rows,before):
                for key in ('instantaneous','cycle_maximum'):
                    self.assertTrue(row[key]['unknown_geometry'].equals(original[key]['unknown_geometry']))
                    self.assertEqual(row[key]['source_fields'][0]['numeric_sha256'],original[key]['source_fields'][0]['numeric_sha256'])

    def test_optional_instantaneous_zoom_keeps_original_domain_and_pressure_scale(self):
        from bevel_gear.contact_composite import plot_contact_composite
        rows=[]
        for method in ('full_gear_fe','local_halfspace'):
            rows.append(dict(method=method,instantaneous=self.build([square([0.,100.,100.,0.])],kind='instantaneous',method=method),
                cycle_maximum=self.build([square([100.]*4)],method=method)))
        before=[[field['numeric_sha256'] for field in row['instantaneous']['source_fields']] for row in rows]
        with tempfile.TemporaryDirectory() as directory:
            record=plot_contact_composite(rows,directory,lang='en',instantaneous_zoom=True,source_compression='gzip')
            self.assertTrue(record['instantaneous_zoom']);self.assertEqual(len(record['instantaneous_zoom_windows']),2)
            self.assertEqual(record['height_mm'],148)
            for zoom in record['instantaneous_zoom_windows']:
                self.assertTrue(zoom['colour_scale_unchanged']);self.assertFalse(zoom['pressure_values_modified'])
                self.assertEqual(zoom['geometry_aspect'],'equal')
                np.testing.assert_allclose(zoom['bounds_width_height_m'],[-.1,-.2,1.1,1.2])
            self.assertEqual(before,[[field['numeric_sha256'] for field in row['instantaneous']['source_fields']] for row in rows])


if __name__=='__main__':unittest.main()
