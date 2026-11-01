"""真实小TET空请求标签→正式档案→独立响应→unknown压力源的回归。

源保留[0,2]而实际压力field仅[0]；空请求不补零、不作材料域认证。
这是明确无CAD的两有限弹性块数值控制，不伪造实际齿轮几何或执行指纹。
"""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
from shapely.geometry import box

from test_native_fe_case import build_fixture, runner, study_inputs
from bevel_gear.native_results_archive import load_native_cycle_source,load_native_phase_from_cycle,audit_native_phase_response
from bevel_gear.native_results_workflow import load_native_workflow_meshes,physical_fixed_nodes
from bevel_gear.native_results_pressure import export_native_pressure_source
from bevel_gear.native_results_external import native_field_partition
from bevel_gear.contact_composite import load_triangle_pressure_map


class EmptyNativePatchArchiveTests(unittest.TestCase):
    def test_actual_empty_request_can_be_omitted_from_result_fields(self):
        explicit=os.environ.get('BEVEL_NATIVE_EMPTY_PATCH_EVIDENCE')
        temporary=None
        if explicit:
            root=Path(explicit).resolve();root.mkdir(parents=True,exist_ok=False)
        else:
            temporary=tempfile.TemporaryDirectory();root=Path(temporary.name)
        prepared=None
        try:
            prepared,template,compact=build_fixture()
            empty=deepcopy(template['patches'][0]);empty['tooth_offset']=2
            empty['requested_mask']=np.zeros_like(empty['requested_mask'],dtype=bool)
            template['patches'].append(empty)
            # 空标签没有物理载荷列；仍给出属于原mesh的真实边界三角和global IDs。
            # 同一合成表面的别名仅用于空请求标签，明确不是另造一枚GEAR齿。
            for member in ('pinion','wheel'):
                part=compact[member][0] if 0 in compact[member] else compact[member]
                compact[member]={0:part,2:deepcopy(part)}
            study=runner.run_grid_study(prepared,[dict(name='one',phase_index=0,template=template)],root/'study',
                workflow_inputs=study_inputs([0.]),compact_facets=compact,solver=dict(batch_cells=32))
            self.assertTrue(study['analysis_completed'])
            workflow=root/'study/one'
            source=load_native_cycle_source(workflow/'distributed_fe_summary.json',require_complete_cycle=False)
            phase=load_native_phase_from_cycle(source,0)
            meshes,files=load_native_workflow_meshes(phase)
            boundary={
                'pinion':dict(kind='plane',axis=2,coordinate_m=-.001,tolerance_m=1e-15),
                'wheel':dict(kind='plane',axis=2,coordinate_m=.001,tolerance_m=1e-15)}
            fixed,contract=physical_fixed_nodes(phase,meshes,boundary_conditions=boundary)
            audited=audit_native_phase_response(phase,meshes=meshes,mesh_files=files,fixed_nodes=fixed)
            result=audited['result'];proof=audited['response_evidence']
            self.assertTrue(proof['passed']);self.assertTrue(proof['independent_Ku_recheck'])
            self.assertEqual(result['selected_tooth_offsets'],[0])
            self.assertEqual([f['tooth_offset'] for f in result['fields']],[0])
            partition=native_field_partition(result)
            self.assertEqual(partition['represented_field_offsets'],[0])
            self.assertEqual(partition['unrepresented_nominal_offsets'],[2])
            patches={p['tooth_offset']:p for p in result['geometry']['patches']}
            self.assertEqual(set(patches),{0,2});self.assertFalse(np.asarray(patches[2]['source_valid_mask']).any())
            self.assertFalse(np.asarray(patches[2]['valid_mask']).any())
            self.assertEqual({c['tooth_offset'] for c in result['cells']},{0})
            field=result['fields'][0];original_force=float(np.asarray(field['forces_n']).sum())
            peak=float(np.asarray(field['pressure_pa']).max());outline=box(-.0005,-.0005,.0005,.0005)
            chart=dict(origin_local_m=[.03,0.,0.],tangent_rows_local=[[1.,0.,0.],[0.,1.,0.]])
            target=root/'actual_unknown_pressure.json.gz'
            data=export_native_pressure_source(audited,tooth_offset=0,outline_geometry=outline,
                levels_pa=np.linspace(0.,peak*1.01,13),domain_manifest=None,pinion_request=None,
                allow_unknown_background=True,cartesian_material_chart=chart,output_file=target)
            self.assertTrue(data['certified_zero_geometry'].is_empty)
            self.assertTrue(data['unknown_geometry'].equals(outline));self.assertEqual(data['source_field_count'],1)
            self.assertFalse(data['original_pressure_values_modified'])
            self.assertAlmostEqual(data['source_identity']['original_P0_plane_force_n'],original_force,places=12)
            reloaded=load_triangle_pressure_map(target,recompute=True)
            self.assertEqual(reloaded['source_peak_pressure_pa'],data['source_peak_pressure_pa'])
            self.assertTrue(reloaded['certified_zero_geometry'].is_empty)
            # 缺field的空标签只能明确拒绝；不能返回一个编造的全零压力源。
            with self.assertRaises((ValueError,KeyError,IndexError)):
                export_native_pressure_source(audited,tooth_offset=2,outline_geometry=outline,
                    levels_pa=np.linspace(0.,peak*1.01,13),domain_manifest=None,pinion_request=None,
                    allow_unknown_background=True,cartesian_material_chart=chart)
            record=dict(passed=True,synthetic_FE_fixture_no_CAD=True,source_patch_offsets=[0,2],actual_result_field_offsets=[0],
                independent_response_passed=True,original_plane_force_n=original_force,source_pressure_peak_pa=peak,
                empty_request_field_fabricated=False,empty_request_zero_domain_certified=False,
                Cartesian_outline_entirely_unknown=True,P1_current_schema_recompute_passed=True,
                pressure_json_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),GEAR_geometry_or_API_fingerprint_fabricated=False,
                complete_cycle_or_CAD_domain_certified=False)
            (root/'regression_receipt.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
        finally:
            if prepared is not None:prepared.systems.clear()
            if temporary is not None:temporary.cleanup()


if __name__=='__main__':unittest.main(verbosity=2)
