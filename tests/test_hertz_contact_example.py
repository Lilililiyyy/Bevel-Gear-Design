"""公开Hertz案例：解析材料/载荷标度及参考长度不能改变实际法向物理量。"""
import importlib.util
from pathlib import Path
import unittest

_READY=all(importlib.util.find_spec(name) for name in ('numpy','scipy','vtk','gmsh','shapely'))


def example():
    project=next(p for p in Path(__file__).resolve().parents if (p/'examples/validate_hertz_contact.py').is_file())
    spec=importlib.util.spec_from_file_location('hertz_validation_example',project/'examples/validate_hertz_contact.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


@unittest.skipUnless(_READY,'需要NumPy/SciPy/VTK/Gmsh/Shapely')
class HertzContactExampleTests(unittest.TestCase):
    def test_large_fans_crossing_contact_region_are_rejected(self):
        import numpy as np
        module=example()
        # 大扇片重心在验证区外，但面本身穿过接触中心；不得用重心筛选漏掉。
        top=dict(nodes_m=np.array([[-.01,-.01,0.],[.01,-.01,0.],[.01,.01,0.],[-.01,.01,0.]]),
            triangles=np.array([[0,1,2],[0,2,3]]))
        with self.assertRaisesRegex(ArithmeticError,'2h'):
            module.verify_surface_resolution(top,.0006,.00005)

    def test_two_material_reference_and_load_scaling(self):
        from bevel_gear.elasticity import ElasticMaterial
        module=example();materials=[ElasticMaterial(210e9,.3),ElasticMaterial(70e9,.33)]
        first=module.hertz_reference(1000.,.02,materials);second=module.hertz_reference(8000.,.02,materials)
        expected=1/((1-.3**2)/210e9+(1-.33**2)/70e9)
        self.assertAlmostEqual(first['effective_modulus_pa']/expected,1.,places=14)
        self.assertAlmostEqual(second['contact_radius_m']/first['contact_radius_m'],2.,places=14)
        self.assertAlmostEqual(second['peak_pressure_pa']/first['peak_pressure_pa'],2.,places=14)
        self.assertAlmostEqual(second['contact_area_m2']/first['contact_area_m2'],4.,places=14)
        self.assertAlmostEqual(second['infinite_body_approach_m']/first['infinite_body_approach_m'],4.,places=14)

    def test_reference_length_is_only_a_generalized_unit_mapping(self):
        import numpy as np
        from bevel_gear.elasticity import ElasticMaterial
        module=example();mesh=module.build_block_mesh(width_m=.02,depth_m=.01,h_m=.0003,
            bulk_m=.002,half_width_m=.0012,refined_depth_m=.001,maximum_nodes=30000)
        material=ElasticMaterial(210e9,.3)
        first=module.solve_hertz_contact(mesh,[material,material],force_n=1000.,radius_m=.02,
            pressure_spacing_m=.0003,half_width_m=.0012,reference_length_m=1.)
        second=module.solve_hertz_contact(mesh,[material,material],force_n=1000.,radius_m=.02,
            pressure_spacing_m=.0003,half_width_m=.0012,reference_length_m=.1)
        self.assertAlmostEqual(first['normal_force_n'],1000.,places=8)
        self.assertAlmostEqual(second['normal_force_n'],1000.,places=8)
        self.assertAlmostEqual(first['normal_approach_m']/second['normal_approach_m'],1.,places=10)
        np.testing.assert_allclose(first['raw_P0']['pressure_pa'],second['raw_P0']['pressure_pa'],rtol=1e-10,atol=.01)
        self.assertFalse(first['halfspace_compliance_used']);self.assertFalse(first['pressure_filter_applied'])
        self.assertAlmostEqual(first['continuous_display']['diagnostics']['original_normal_force_n'],1000.,places=8)


if __name__=='__main__':unittest.main()
