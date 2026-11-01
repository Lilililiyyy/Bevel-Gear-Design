"""合法混合维面域的发布回归；不建立CAD、FE系统或重采样压力。"""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import shapely
from shapely.geometry import GeometryCollection,LineString,Point,box

from bevel_gear.contact_composite import (_area_geometry,build_triangle_pressure_map,
    export_triangle_pressure_map,load_triangle_pressure_map)


class MixedAreaGeometryTests(unittest.TestCase):
    def test_mixed_zero_domain_keeps_holes_area_pressure_and_roundtrip(self):
        outline=box(0.,0.,.004,.004);support=box(.0015,0.,.0025,.004);unknown=box(.0003,.002,.0008,.003)
        zero=outline.difference(support).difference(unknown)
        # 集合运算可保留共边/孤立线和点；它们不构成任何新增二维零压域。
        polygons=list(zero.geoms)[::-1]
        raw_zero=GeometryCollection([GeometryCollection(polygons),LineString([(.0004,.0021),(.0007,.0029)]),Point(.0005,.0025)])
        raw_unknown=GeometryCollection([unknown,LineString([(0.,.004),(.004,.004)])])
        original_wkb=shapely.to_wkb(raw_zero);raw_area=raw_zero.area
        cleaned=_area_geometry(raw_zero,'有证据零压域')
        self.assertEqual(cleaned.area,raw_area);self.assertEqual(cleaned.symmetric_difference(raw_zero).area,0.)
        self.assertEqual(cleaned.symmetric_difference(zero).area,0.);self.assertFalse(cleaned.covers(Point(.0005,.0025)))
        self.assertEqual([part.wkb for part in cleaned.geoms],[part.wkb for part in polygons])
        self.assertEqual(shapely.to_wkb(raw_zero),original_wkb)
        line=LineString([(0.,0.),(.004,.004)])
        self.assertTrue(_area_geometry(line,'空零压面域').is_empty)
        self.assertTrue(_area_geometry(Point(0.,0.),'空未知面域').is_empty)
        with self.assertRaises(ValueError):_area_geometry(line,'实际齿面轮廓',allow_empty=False)
        field=dict(points_width_height_m=np.array([[.0015,0.],[.0025,0.],[.0025,.004],[.0015,.004]]),
            triangles=np.array([[0,1,2],[0,2,3]]),pressure_pa=np.array([100e6,400e6,500e6,200e6]),phase_index=0,tooth_offset=0)
        pressure_before=field['pressure_pa'].copy()
        data=build_triangle_pressure_map([field],method='full_gear_fe',kind='instantaneous',outline_geometry=outline,
            levels_pa=[0.,200e6,400e6,600e6],unknown_geometry=raw_unknown,certified_zero_geometry=raw_zero)
        self.assertEqual(data['certified_zero_geometry'].area,zero.area)
        self.assertEqual(data['unknown_geometry'].area,unknown.area)
        self.assertEqual(data['certified_zero_geometry'].symmetric_difference(zero).area,0.)
        self.assertFalse(data['original_pressure_values_modified']);np.testing.assert_array_equal(field['pressure_pa'],pressure_before)
        with tempfile.TemporaryDirectory() as folder:
            target=Path(folder)/'mixed_pressure_map.json.gz';export_triangle_pressure_map(data,target)
            restored=load_triangle_pressure_map(target,recompute=True)
        self.assertEqual(restored['certified_zero_geometry'].symmetric_difference(zero).area,0.)
        self.assertEqual(restored['unknown_geometry'].symmetric_difference(unknown).area,0.)
        np.testing.assert_array_equal(restored['source_fields'][0]['pressure_pa'],pressure_before)
        self.assertEqual(shapely.to_wkb(raw_zero),original_wkb)


    def test_machine_collinear_zero_polygon_normalizes_without_repair(self):
        from shapely.geometry import Polygon,LinearRing
        # 来自实际派生差集的有限共线舍入情形：GEOSarea=0，hull仅机器尺度。
        raw=Polygon([(1.0491607582707729e-13,.0020106954692348175),
            (-6.855627141291385e-15,.0018478872729415065),
            (-6.855627177060342e-15,.001847887272889405)])
        self.assertTrue(raw.is_valid);self.assertFalse(raw.is_empty);self.assertEqual(raw.area,0.)
        self.assertGreater(raw.convex_hull.area,0.)
        before=raw.wkb;empty=_area_geometry(raw,'派生未分类域')
        self.assertTrue(empty.is_empty);self.assertEqual(empty.symmetric_difference(raw).area,0.)
        self.assertEqual(raw.wkb,before)
        with self.assertRaises(ValueError):_area_geometry(raw,'实际工作面轮廓',allow_empty=False)
        bowtie=Polygon([(0.,0.),(.001,.001),(0.,.001),(.001,0.)])
        self.assertEqual(bowtie.area,0.);self.assertFalse(bowtie.is_valid);self.assertGreater(bowtie.convex_hull.area,0.)
        with self.assertRaises(ValueError):_area_geometry(bowtie,'自交面不得删除')
        # 拓扑维度为1的弯折线/ring没有面域，正凸包不改变这一事实。
        bent=LineString([(0.,0.),(.001,.001),(.002,0.)])
        ring=LinearRing([(0.,0.),(.001,0.),(.001,.001),(0.,.001),(0.,0.)])
        for line in (bent,ring):
            self.assertTrue(line.is_valid);self.assertEqual(line.area,0.);self.assertGreater(line.convex_hull.area,0.)
            wkb=line.wkb;self.assertTrue(_area_geometry(line,'合法一维残片').is_empty)
            self.assertEqual(line.wkb,wkb)
            with self.assertRaises(ValueError):_area_geometry(line,'实际工作面轮廓',allow_empty=False)
        outline=box(0.,0.,.004,.004);support=box(0.,0.,.001,.001)
        field=dict(points_width_height_m=np.array([[0.,0.],[.001,0.],[.001,.001],[0.,.001]]),
            triangles=np.array([[0,1,2],[0,2,3]]),pressure_pa=np.array([100e6,400e6,500e6,200e6]),phase_index=0,tooth_offset=0)
        data=build_triangle_pressure_map([field],method='full_gear_fe',kind='instantaneous',outline_geometry=outline,
            levels_pa=[0.,200e6,400e6,600e6],unknown_geometry=GeometryCollection(),certified_zero_geometry=outline.difference(support))
        data['unclassified_geometry']=raw  # 合法零测度派生差集，不是新增承载域。
        with tempfile.TemporaryDirectory() as folder:
            target=Path(folder)/'zero_area_map.json.gz';export_triangle_pressure_map(data,target)
            restored=load_triangle_pressure_map(target)
        self.assertTrue(restored['unclassified_geometry'].is_empty)
        np.testing.assert_array_equal(restored['source_fields'][0]['pressure_pa'],field['pressure_pa'])
        self.assertEqual(raw.wkb,before)


if __name__=='__main__':unittest.main()
