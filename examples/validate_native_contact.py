"""两块真实TET4弹性体的原生FE接触轻量验证，不使用模拟QP。

下块的实际节点构成凸抛物面，上块为平面；间隙由真实FE边界求交得到。
调用正式solve_native_fe_contact，独立稠密消元复核最终活动柔度、位移、
能量及轴矩。它是数值接口回归，不是齿轮算例或Hertz无限体误差认证。

运行：python examples/validate_native_contact.py
依赖：NumPy、SciPy、Shapely；不需要CAD引擎、网格器或厂商组件。
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

PROJECT=next((p for p in Path(__file__).resolve().parents
    if (p/'bevel_gear').is_dir() and (p/'pyproject.toml').is_file()),None)
if PROJECT is not None:sys.path.insert(0,str(PROJECT))


def _identity(value):
    """绑定原数值数组；仅用于本案例前后字节复核。"""
    import numpy as np
    a=np.ascontiguousarray(value)
    return dict(shape=list(a.shape),dtype=a.dtype.name,
        sha256=hashlib.sha256(memoryview(a).cast('B')).hexdigest())


def _block(member,*,divisions,width_m,depth_m,centre_x_m,radius_m):
    """一致六四面体划分；完整外边界由单邻接面产生并朝实体外定向。"""
    import numpy as np
    from bevel_gear.elasticity import TetraMesh
    n=divisions;layers=2;nodes=[]
    for k in range(layers+1):
        for j in range(n+1):
            for i in range(n+1):
                x=width_m*(i/n-.5);y=width_m*(j/n-.5)
                if member=='pinion':
                    surface=-(x*x+y*y)/(2*radius_m)
                    z=-depth_m+(k/layers)*(surface+depth_m)
                else:z=depth_m*k/layers
                nodes.append((centre_x_m+x,y,z))
    nodes=np.asarray(nodes)
    def index(i,j,k):return (k*(n+1)+j)*(n+1)+i
    tetra=[]
    for k in range(layers):
        for j in range(n):
            for i in range(n):
                for order in itertools.permutations(range(3)):
                    q=np.array((i,j,k));ids=[index(*q)]
                    for axis in order:
                        q=q.copy();q[axis]+=1;ids.append(index(*q))
                    tetra.append(ids)
    tetra=np.asarray(tetra,dtype=np.int64)
    adjacency={}
    for owner,tet in enumerate(tetra):
        for face in ((1,2,3),(0,2,3),(0,1,3),(0,1,2)):
            tri=tuple(int(tet[q]) for q in face);key=tuple(sorted(tri))
            adjacency.setdefault(key,[]).append((tri,owner))
    if any(len(value)>2 for value in adjacency.values()):raise ArithmeticError('合成四面体出现非流形面')
    faces=[]
    for value in adjacency.values():
        if len(value)!=1:continue
        tri,owner=value[0];tri=list(tri);xyz=nodes[tri]
        if np.cross(xyz[1]-xyz[0],xyz[2]-xyz[0])@(xyz.mean(axis=0)-nodes[tetra[owner]].mean(axis=0))<0:
            tri[1],tri[2]=tri[2],tri[1]
        faces.append(tri)
    faces=np.asarray(faces,dtype=np.int64)
    support_layer=0 if member=='pinion' else layers
    contact_layer=layers if member=='pinion' else 0
    fixed=np.asarray([index(i,j,support_layer) for j in range(n+1) for i in range(n+1)],dtype=np.int64)
    surface_ids=np.asarray([index(i,j,contact_layer) for j in range(n+1) for i in range(n+1)],dtype=np.int64)
    selected=np.isin(faces,surface_ids).all(axis=1);top=faces[selected]
    inverse=np.full(len(nodes),-1,dtype=np.int64);inverse[surface_ids]=np.arange(len(surface_ids))
    compact=dict(name=member,nodes_m=nodes[surface_ids].copy(),triangles=inverse[top],
        global_node_ids=surface_ids,total_node_count=len(nodes))
    mesh=TetraMesh(nodes,tetra,faces,np.zeros(len(faces),dtype=np.int64),
        dict(units='m',geometry='合成凸抛物面/平面有限弹性块',boundary='完整单邻接tet外边界'))
    return mesh,fixed,compact


def build_fixture(*,divisions=7,pressure_count=8,torque_nm=-.003):
    """构造两块真实ElasticSystem；远离接触的端面全约束，三个平移为零。"""
    import numpy as np
    from bevel_gear.elasticity import ElasticMaterial,ElasticSystem
    if isinstance(divisions,bool) or not isinstance(divisions,int) or not 6<=divisions<=10:
        raise ValueError('轻量案例的divisions须为6至10整数')
    if isinstance(pressure_count,bool) or not isinstance(pressure_count,int) or not 7<=pressure_count<=11:
        raise ValueError('压力格数须为7至11整数')
    if not math.isfinite(torque_nm) or not -.03<=torque_nm<0:
        raise ValueError('本案例负扭矩须满足−0.03≤T<0 N·m')
    width=.001;depth=.001;radius=.1;centre=.03
    material=ElasticMaterial(210e9,.3);systems={};compact={}
    for member in ('pinion','wheel'):
        mesh,fixed,part=_block(member,divisions=divisions,width_m=width,depth_m=depth,
            centre_x_m=centre,radius_m=radius)
        systems[member]=ElasticSystem(mesh,material,fixed,maximum_dofs=1200)
        compact[member]={0:part}
    if sum(len(s.mesh.nodes_m) for s in systems.values())>=1000:
        raise ValueError('此回归只允许两体合计少于1000节点')
    prepared=SimpleNamespace(systems=systems,
        models={m:SimpleNamespace(request=SimpleNamespace(tooth_count=17)) for m in systems})
    frames=dict(pinion_rotation_rows=np.eye(3),wheel_rotation_rows=np.eye(3),
        pinion_axis_world=[0.,1.,0.],wheel_axis_world=[0.,-1.,0.],translation_m=[0.,0.,0.])
    spacing=.0008/(pressure_count-1)
    template=dict(frames=frames,driving_angle_rad=0.,reference_follower_angle_rad=0.,
        pinion_torque_nm=float(torque_nm),pinion_to_wheel_ratio=1.,patches=[dict(tooth_offset=0,
            grid=dict(x_count=pressure_count,y_count=pressure_count,spacing_x_m=spacing,spacing_y_m=spacing,
                origin_xy_m=(-.0004,-.0004)),
            plane=dict(origin_world_m=[centre,0.,0.],tangent_rows=[[1.,0.,0.],[0.,1.,0.]],normal_world=[0.,0.,1.]),
            requested_mask=np.ones((pressure_count,pressure_count),dtype=bool),source_material_reference_frames=deepcopy(frames))])
    return prepared,template,compact


def audit_dense_reference(prepared,answer,template):
    """独立dense Kff消元；不调用原QP，也不以理论Hertz间隙替换FE几何。"""
    import numpy as np
    from scipy.linalg import cho_factor,cho_solve
    core=answer['result'];system=answer['system'];forces=np.asarray(core['equilibrium']['forces_n'])
    active=np.asarray(core['equilibrium']['active_indices'],dtype=int)
    records=system['records'];gaps=np.asarray([r['gap_m'] for r in records])
    ell=np.asarray([r['levers_m']['wheel'] for r in records]);delta=core['linearization_step_theta_rad']
    ca=np.zeros((len(active),len(active)));sparse_ca=np.zeros_like(ca);body_rows={};energy=0.;memory=0
    for member,body in prepared.systems.items():
        W=system['matrices'][member];free=body.free_dofs;Wa=W[free][:,active].toarray()
        Kff=body.stiffness[free][:,free].toarray()
        # 新的独立稠密Cholesky，仅使用同一已组装Kff，不复用原稀疏因子。
        dense_factor=cho_factor(Kff,lower=True,check_finite=True)
        columns=cho_solve(dense_factor,Wa)
        ca+=Wa.T@columns;sparse_ca+=Wa.T@body.factorization.solve(Wa)
        expected_force=np.asarray(W@forces).ravel()
        expected_u=np.zeros_like(expected_force)
        expected_u[free]=cho_solve(dense_factor,expected_force[free])
        original=core['body_fields'][member]
        original_u=np.asarray(original['displacements_m']).ravel();original_F=np.asarray(original['nodal_forces_n']).ravel()
        internal=body.stiffness@original_u
        relative_residual=np.linalg.norm((internal-expected_force)[free])/max(np.linalg.norm(expected_force[free]),1e-30)
        displacement_difference=np.linalg.norm(original_u-expected_u)/max(np.linalg.norm(expected_u),1e-30)
        load_difference=np.linalg.norm(original_F-expected_force)/max(np.linalg.norm(expected_force),1e-30)
        actual_energy=.5*float(original_u@internal);work=float(original_u@expected_force);energy+=actual_energy
        R=np.asarray(system['frames'][member+'_rotation_rows']);axis=np.asarray(system['frames'][member+'_axis_world'])
        world=np.asarray(body.mesh.nodes_m)@R.T;worldforce=expected_force.reshape(-1,3)@R.T
        # 轴矩为接触节点力矩；原QP的ell采用平衡外矩，故为此量的相反数。
        contact_moment=float(np.cross(world,worldforce).sum(axis=0)@axis)
        resisting_torque=-contact_moment
        record_torque=float(np.asarray([r['levers_m'][member] for r in records])@forces)
        memory+=Kff.nbytes+Wa.nbytes+columns.nbytes+sum(a.nbytes for a in
            (body.nodes,body.tetrahedra,body.b,body.volumes,body.dofs))
        body_rows[member]=dict(nodes=len(body.mesh.nodes_m),tetrahedra=len(body.mesh.tetrahedra),free_dofs=len(free),
            dense_vs_sparse_displacement_relative_error=float(displacement_difference),
            original_Wf_relative_error=float(load_difference),free_Ku_minus_F_relative_error=float(relative_residual),
            fixed_displacement_maximum_m=float(np.max(np.abs(original_u[body.fixed_dofs]))),
            strain_energy_j=actual_energy,external_work_j=work,
            energy_work_relative_error=abs(2*actual_energy-work)/max(abs(work),1e-30),
            contact_axis_moment_nm=contact_moment,reconstructed_external_torque_nm=resisting_torque,
            original_record_torque_nm=record_torque,
            nodal_vs_record_torque_relative_error=abs(resisting_torque-record_torque)/max(abs(record_torque),1e-30))
    sparse_error=np.linalg.norm(ca-sparse_ca)/max(np.linalg.norm(ca),1e-30)
    f=forces[active];aell=ell[active];agap=gaps[active]
    z=np.linalg.solve(ca,aell);v=np.linalg.solve(ca,agap)
    target=template['pinion_torque_nm']/template['pinion_to_wheel_ratio']
    dense_delta=(target+aell@v)/(aell@z);dense_f=z*dense_delta-v
    stationarity=agap+ca@f-aell*delta
    stiffness=float(aell@z);condensed=.5*float(f@ca@f)
    final_TE=template['reference_follower_angle_rad']+sum(h['delta_theta_rad'] for h in answer['geometry_outer_history'])+template['pinion_to_wheel_ratio']*template['driving_angle_rad']
    field=core['fields'][0];pressure=np.asarray(field['pressure_pa']);mask=np.asarray(field['contact_mask'])
    boundary=np.r_[mask[0],mask[-1],mask[:,0],mask[:,-1]]
    result=dict(active_count=len(active),candidate_count=len(forces),zero_pressure_count=int((pressure==0).sum()),
        original_active_force_n=f.tolist(),
        active_C_dense_vs_sparse_relative_error=float(sparse_error),
        active_C_symmetry_relative_error=float(np.linalg.norm(ca-ca.T)/max(np.linalg.norm(ca),1e-30)),
        active_C_minimum_eigenvalue_m_per_n=float(np.linalg.eigvalsh(ca).min()),
        final_active_stationarity_maximum_m=float(np.max(np.abs(stationarity))),
        dense_active_force_relative_error=float(np.linalg.norm(dense_f-f)/max(np.linalg.norm(f),1e-30)),
        dense_last_delta_absolute_error_rad=abs(float(dense_delta)-delta),
        independent_incremental_stiffness_nm_per_rad=stiffness,
        reported_stiffness_relative_error=abs(stiffness-core['incremental_stiffness_nm_per_rad'])/max(abs(stiffness),1e-30),
        FE_vs_dense_condensed_energy_relative_error=abs(energy-condensed)/max(abs(condensed),1e-30),
        QP_vs_dense_condensed_energy_relative_error=abs(core['equilibrium']['strain_energy_j']-condensed)/max(abs(condensed),1e-30),
        independent_total_loaded_TE_rad=float(final_TE),reported_total_TE_absolute_error_rad=abs(final_TE-core['total_loaded_te_rad']),
        loaded_pressure_window_has_zero_neighbor_ring=bool(not boundary.any() and (pressure==0).any()),
        actual_last_delta_rad=float(delta),actual_total_loaded_TE_rad=core['total_loaded_te_rad'],
        body_checks=body_rows,active_C_m_per_n=ca.tolist(),independent_reference_working_arrays_bytes=memory,
        all_pressure_values_from_original_FE_QP=True,halfspace_or_Hertz_gap_used=False)
    passed=(sparse_error<1e-10 and result['active_C_minimum_eigenvalue_m_per_n']>0
        and result['final_active_stationarity_maximum_m']<=1e-10
        and result['dense_active_force_relative_error']<1e-8 and result['dense_last_delta_absolute_error_rad']<1e-12
        and result['reported_stiffness_relative_error']<1e-9
        and result['FE_vs_dense_condensed_energy_relative_error']<1e-9
        and result['QP_vs_dense_condensed_energy_relative_error']<1e-9
        and result['reported_total_TE_absolute_error_rad']<1e-14
        and result['loaded_pressure_window_has_zero_neighbor_ring']
        and all(max(row['dense_vs_sparse_displacement_relative_error'],row['original_Wf_relative_error'],
            row['free_Ku_minus_F_relative_error'],row['energy_work_relative_error'],row['nodal_vs_record_torque_relative_error'])<1e-9
            and row['fixed_displacement_maximum_m']==0. for row in body_rows.values()))
    result['passed']=bool(passed)
    return result


def run_validation(*,divisions=7,pressure_count=8,torque_nm=-.003,progress=None):
    """正式原生FE+真实QP端到端验证；失败保留原拒绝原因，不放宽门限。"""
    import numpy as np
    from bevel_gear.native_fe_contact import solve_native_fe_contact
    prepared,template,compact=build_fixture(divisions=divisions,pressure_count=pressure_count,torque_nm=torque_nm)
    before={m:dict(nodes=_identity(s.mesh.nodes_m),tetrahedra=_identity(s.mesh.tetrahedra),
        boundary_triangles=_identity(s.mesh.boundary_triangles),boundary_tags=_identity(s.mesh.boundary_tags))
        for m,s in prepared.systems.items()}
    answer=solve_native_fe_contact(prepared,template,compact_facets=compact,maximum_outer_iterations=6,
        outer_step_tolerance_rad=1e-8,mean_geometry_tolerance_m=1e-10,maximum_piece_penetration_m=2e-7,
        maximum_fe_ray_closure_m=1e-10,maximum_working_set=256,batch_cells=32,include_body_fields=True,progress=progress)
    audit=None if not answer['accepted'] else audit_dense_reference(prepared,answer,template)
    after={m:dict(nodes=_identity(s.mesh.nodes_m),tetrahedra=_identity(s.mesh.tetrahedra),
        boundary_triangles=_identity(s.mesh.boundary_triangles),boundary_tags=_identity(s.mesh.boundary_tags))
        for m,s in prepared.systems.items()}
    if before!=after:raise ArithmeticError('原生接触验证改写了输入网格')
    core=answer['result']
    return dict(schema='small_true_TET4_native_contact_validation_v1',
        passed=bool(answer['accepted'] and audit and audit['passed']),native_contact_accepted=answer['accepted'],
        rejection_reasons=answer['rejection_reasons'],dense_reference=audit,
        geometry_check=answer['geometry_check'],geometry_outer_history=answer['geometry_outer_history'],
        geometry_source='实际TET4外边界，凸抛物面节点/平面节点；没有解析gap输入',
        input=dict(divisions=divisions,pressure_count=pressure_count,torque_nm=torque_nm,
            width_m=.001,depth_m=.001,curvature_radius_m=.1,axis_offset_m=.03,young_modulus_pa=210e9,poisson_ratio=.3,
            pinion_axis_world=[0.,1.,0.],wheel_axis_world=[0.,-1.,0.],tooth_counts=[17,17]),
        actual_mesh_identity=before,input_mesh_arrays_unchanged=True,
        preparation_audit=answer['preparation_audit'],input_fingerprint=answer['input_fingerprint'],
        pressure_pa=None if core is None or 'fields' not in core else np.asarray(core['fields'][0]['pressure_pa']).tolist(),
        status=None if core is None else core.get('status'),
        kernel='bevel_gear.native_fe_contact.solve_native_fe_contact，原生产QP及完整几何门控',
        limitations=['合成有限体的轻量数值回归，不是实际齿轮几何或Hertz无限体误差认证',
            '只检查指定有限候选窗，未认证其他材料面/大滑动/体网格及压力网格收敛',
            '总受载TE与末步delta分开；未求卸载FE网格相切，不宣称纯弹性TE'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--divisions',type=int,default=7)
    parser.add_argument('--pressure-count',type=int,default=8)
    parser.add_argument('--torque-nm',type=float,default=-.003)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    result=run_validation(divisions=args.divisions,pressure_count=args.pressure_count,torque_nm=args.torque_nm)
    if args.output is not None:
        if args.output.exists():raise FileExistsError('不能覆盖已有验证输出')
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,ensure_ascii=False,allow_nan=False,indent=2,
            default=lambda value:value.tolist())+'\n',encoding='utf8')
    print(json.dumps(dict(passed=result['passed'],native_contact_accepted=result['native_contact_accepted'],
        rejection_reasons=result['rejection_reasons'],outer_iterations=len(result['geometry_outer_history']),
        dense_reference=result['dense_reference']),ensure_ascii=False),flush=True)
    if not result['passed']:raise SystemExit('原生FE端到端验证未通过，保留真实失败状态')


if __name__=='__main__':main()
