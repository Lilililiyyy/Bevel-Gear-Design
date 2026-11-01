"""可冷启动的Hertz物理基准：二次初隙、两有限弹性块体和真实P0分布牵引。

默认h=50µm、压力步长100µm、4×4积分。SciPy直接分解可能需要数GB内存；
可用--mesh-um 150快速运行粗网格，但粗结果必须保留实际验证失败状态。
该模型不是球形实体：两平面FE迹面以Hertz小斜率二次初隙相对接近。
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import argparse
import hashlib
import json
import math
import sys

PROJECT=next((p for p in Path(__file__).resolve().parents if (p/'bevel_gear').is_dir() and (p/'pyproject.toml').is_file()),None)
if PROJECT is not None:sys.path.insert(0,str(PROJECT))


def hertz_reference(force_n,radius_m,materials):
    """两材料经典Hertz圆接触，E*为约化模量Pa，R为相对曲率半径m。"""
    from bevel_gear.elasticity import ElasticMaterial
    if not math.isfinite(force_n) or force_n<=0 or not math.isfinite(radius_m) or radius_m<=0:
        raise ValueError('法向力N和相对曲率半径m须为正有限数')
    materials=[m if isinstance(m,ElasticMaterial) else ElasticMaterial(**m) for m in materials]
    if len(materials)!=2:raise ValueError('须给出两个弹性体的材料')
    effective=1/sum((1-m.poisson_ratio**2)/m.young_modulus_pa for m in materials)
    a=(3*force_n*radius_m/(4*effective))**(1/3)
    return dict(force_n=force_n,relative_radius_m=radius_m,effective_modulus_pa=effective,
        contact_radius_m=a,peak_pressure_pa=3*force_n/(2*math.pi*a*a),contact_area_m2=math.pi*a*a,
        infinite_body_approach_m=a*a/radius_m)


def build_block_mesh(*,width_m,depth_m,h_m,bulk_m,half_width_m,refined_depth_m,maximum_nodes):
    """Gmsh/OpenCASCADE实体与嵌入顶面点；实际接触区不依赖单点尺寸传播。"""
    import gmsh
    import numpy as np
    from bevel_gear.elasticity import TetraMesh
    if not 0<h_m<=bulk_m or half_width_m*4/3>=width_m/2 or not h_m<=refined_depth_m<depth_m:
        raise ValueError('块体/局部网格尺寸或接触窗口非法')
    gmsh.initialize(['-nopopup']);gmsh.option.setNumber('General.Terminal',0)
    try:
        gmsh.model.add('hertz_block')
        gmsh.model.occ.addBox(-width_m/2,-width_m/2,-depth_m,width_m,width_m,depth_m)
        axis=np.linspace(-half_width_m,half_width_m,math.ceil(2*half_width_m/h_m)+1)
        seeds=[gmsh.model.occ.addPoint(float(x),float(y),0.,h_m) for y in axis for x in axis]
        # 整个近表层体积加密；仅顶面细、第一层之后立即变粗会高估接触刚度。
        reach=half_width_m*4/3
        rows=np.linspace(-reach,reach,math.ceil(2*reach/h_m)+1)
        depths=np.linspace(-refined_depth_m,0.,math.ceil(refined_depth_m/h_m)+1)
        tracks=[]
        for z in depths:
            for y in rows:
                left=gmsh.model.occ.addPoint(-reach,float(y),float(z))
                right=gmsh.model.occ.addPoint(reach,float(y),float(z))
                tracks.append(gmsh.model.occ.addLine(left,right))
        gmsh.model.occ.synchronize()
        top=[tag for dim,tag in gmsh.model.getEntities(2) if abs(gmsh.model.occ.getCenterOfMass(dim,tag)[2])<1e-12]
        if len(top)!=1:raise ArithmeticError('顶面实体不唯一')
        gmsh.model.mesh.embed(0,seeds,2,top[0])
        distance=gmsh.model.mesh.field.add('Distance');gmsh.model.mesh.field.setNumbers(distance,'CurvesList',tracks)
        gmsh.model.mesh.field.setNumber(distance,'Sampling',100)
        threshold=gmsh.model.mesh.field.add('Threshold')
        for name,value in dict(InField=distance,SizeMin=h_m,SizeMax=bulk_m,DistMin=h_m,DistMax=max(.001,4*h_m)).items():
            gmsh.model.mesh.field.setNumber(threshold,name,value)
        gmsh.model.mesh.field.setAsBackgroundMesh(threshold)
        gmsh.option.setNumber('Mesh.MeshSizeMin',h_m);gmsh.option.setNumber('Mesh.MeshSizeMax',bulk_m)
        gmsh.option.setNumber('Mesh.MeshSizeExtendFromBoundary',0)
        gmsh.option.setNumber('Mesh.MeshSizeFromPoints',0);gmsh.option.setNumber('Mesh.MeshSizeFromCurvature',0)
        gmsh.option.setNumber('Mesh.ElementOrder',1);gmsh.option.setNumber('Mesh.Algorithm3D',1)
        gmsh.option.setNumber('General.NumThreads',1);gmsh.model.mesh.generate(3)
        tags,coordinates,_=gmsh.model.mesh.getNodes();nodes=np.asarray(coordinates).reshape((-1,3))
        lookup={int(tag):i for i,tag in enumerate(tags)};types,_,all_nodes=gmsh.model.mesh.getElements(3)
        tets=[np.asarray([lookup[int(n)] for n in raw]).reshape((-1,4)) for typ,raw in zip(types,all_nodes) if typ==4]
        if not tets:raise ArithmeticError('未生成四节点实体四面体')
        tetra=np.concatenate(tets);used,inverse=np.unique(tetra,return_inverse=True);nodes=nodes[used];tetra=inverse.reshape((-1,4))
        if len(nodes)>maximum_nodes:raise ValueError('实体网格超出显式节点上限；请增大h或maximum_nodes')
        return TetraMesh(nodes,tetra,metadata=dict(units='m',mesher='Gmsh_OpenCASCADE',
            block_width_m=width_m,block_depth_m=depth_m,local_size_m=h_m,bulk_size_m=bulk_m,
            embedded_top_seed_count=len(seeds),contact_half_width_m=half_width_m,
            refined_volume_half_width_m=reach,refined_depth_m=refined_depth_m,refinement_curve_count=len(tracks)))
    finally:gmsh.finalize()


def verify_surface_resolution(top,half_width_m,h_m):
    """检查所有与接触包围正方形相交的实际三角面，禁止重心筛选漏掉扇片。"""
    import numpy as np
    import shapely
    from shapely.geometry import box
    vertices=np.asarray(top['nodes_m'])[np.asarray(top['triangles'])]
    polygons=shapely.polygons(vertices[:,:,:2]);region=box(-half_width_m,-half_width_m,half_width_m,half_width_m)
    ids=shapely.STRtree(polygons).query(region,predicate='intersects')
    if not len(ids):raise ArithmeticError('Hertz接触包围域没有实际FE三角形')
    edges=np.linalg.norm(vertices[ids]-np.roll(vertices[ids],1,axis=1),axis=2)
    overlap=shapely.area(shapely.intersection(polygons[ids],region))
    report=dict(region_half_width_m=half_width_m,triangle_count=len(ids),maximum_edge_m=float(edges.max()),
        edge_p95_m=float(np.quantile(edges,.95)),partition_relative_area_error=abs(float(overlap.sum()/region.area)-1),
        requested_h_m=h_m,selection='整三角与GEOS矩形相交，包含细区内伸入的粗扇片')
    if report['maximum_edge_m']>2*h_m or report['partition_relative_area_error']>1e-9:
        raise ArithmeticError('实际接触区面网格不满足2h上限或覆盖不完整，拒绝进入弹性求解')
    return report


def solve_hertz_contact(mesh,materials,*,force_n,radius_m,pressure_spacing_m,half_width_m,
        quadrature_order=4,maximum_dofs=900000,factorization_factory=None,reference_length_m=1.):
    """同一W构造载荷和位移投影；返回法向F、接近量δ，不暴露虚拟齿轮模型。

    原接触QP以广义量工作：T=F L0、δ=L0 θ，L0=1m为单位映射。
    物理柔度仍是ΣW.T Kff^-1 W，未使用半空间柔度或解析压力加载。
    """
    import numpy as np
    from scipy import sparse
    from bevel_gear.elasticity import ElasticSystem,ElasticMaterial
    from bevel_gear.contact_mesh import boundary_facets
    from bevel_gear.generated_distributed_fe import _SurfaceTraceLocator
    from bevel_gear.fe_contact_solver import solve_fe_contact_equilibrium
    from bevel_gear.halfspace_contact import RectangularContactGrid
    from bevel_gear.pressure_field import reconstruct_distributed_fe_pressure_field
    if isinstance(quadrature_order,bool) or not isinstance(quadrature_order,int) or not 2<=quadrature_order<=12:
        raise ValueError('Gauss阶数须为2至12的整数')
    if pressure_spacing_m<=0 or reference_length_m<=0:raise ValueError('压力步长和参考长度须为正')
    nodes=np.asarray(mesh.nodes_m);bottom=float(nodes[:,2].min());fixed=np.flatnonzero(abs(nodes[:,2]-bottom)<1e-12)
    facets=boundary_facets(mesh);keep=(facets['normals'][:,2]>.99999)&np.all(abs(nodes[facets['triangles']][:,:,2])<1e-10,axis=1)
    top={k:v if k in ('nodes_m','metadata') else v[keep] for k,v in facets.items()}
    materials=[m if isinstance(m,ElasticMaterial) else ElasticMaterial(**m) for m in materials]
    bodies=[]
    for i,material in enumerate(materials):
        bodies.append(bodies[0] if i==1 and material==materials[0] else
            ElasticSystem(mesh,material,fixed,maximum_dofs=maximum_dofs,factorization_factory=factorization_factory))
    locator=_SurfaceTraceLocator(top,np.eye(3))
    n=2*round(2*half_width_m/pressure_spacing_m/2)
    if n<4 or n*n>30000:raise ValueError('压力网格须至少4×4且不超过30000面元')
    step=2*half_width_m/n;area=step**2;axis=(np.arange(n)+.5)*step-half_width_m
    xx,yy=np.meshgrid(axis,axis);centers=np.stack((xx,yy,np.zeros_like(xx)),axis=-1)
    gauss,weights=np.polynomial.legendre.leggauss(quadrature_order)
    shifts=np.array([[a*step/2,b*step/2,0.] for b in gauss for a in gauss])
    qweights=np.array([a*b/4 for b in weights for a in weights]);nq=len(qweights)
    points=(centers[:,:,None,:]+shifts).reshape((-1,3))
    mappings,reasons=locator.locate_many(points,np.tile([0.,0.,1.],(len(points),1)),tolerance_m=1e-9)
    if any(reason is not None for reason in reasons):raise ArithmeticError('实际顶面FE形函数定位失败')
    ids=np.asarray([m['nodes'] for m in mappings]);bary=np.asarray([m['weights'] for m in mappings])
    columns=np.repeat(np.arange(n*n),nq*3);rows=(3*ids+2).ravel()
    values=(-np.tile(qweights,n*n)[:,None]*bary).ravel()
    W=sparse.coo_matrix((values,(rows,columns)),shape=(3*len(nodes),n*n)).tocsc()
    # 二次初隙的矩形均值可精确解析积分，不从解析Hertz压力取得任何载荷。
    gaps=(xx.ravel()**2+yy.ravel()**2+step**2/6)/(2*radius_m)
    free=[W[body.free_dofs] for body in bodies];cache={}
    def compliance(indices):
        indices=list(map(int,indices));missing=[i for i in indices if i not in cache]
        if missing:
            result=np.zeros((n*n,len(missing)))
            for body,w in zip(bodies,free):
                for start in range(0,len(missing),24):
                    selected=missing[start:start+24];u=body.factorization.solve(w[:,selected].toarray())
                    result[:,start:start+len(selected)]+=w.T@u
            for j,index in enumerate(missing):cache[index]=result[:,j]
        return np.column_stack([cache[i] for i in indices])
    eq=solve_fe_contact_equilibrium(gaps,lever_arms_m=np.full(n*n,reference_length_m),
        torque_nm=force_n*reference_length_m,compliance_columns=compliance,maximum_working_set=2048,
        gap_tolerance_m=1e-10,relative_tolerance=1e-9)
    if not eq['converged']:raise ArithmeticError('Hertz离散互补求解未接受：'+eq['status'])
    forces=np.asarray(eq['forces_n']);pressure=forces/area;approach=reference_length_m*eq['theta_rad']
    response=[];point_gap=np.einsum('ij,ij->i',points[:,:2],points[:,:2])/(2*radius_m)-approach
    for body in bodies:
        load=np.asarray(W@forces).ravel();u=np.zeros(len(load));u[body.free_dofs]=body.factorization.solve(load[body.free_dofs])
        internal=np.asarray(body.stiffness@u).ravel();relative=float(np.linalg.norm((internal-load)[body.free_dofs])/max(np.linalg.norm(load[body.free_dofs]),1e-30))
        if relative>1e-8:raise ArithmeticError('完整FE响应未通过原刚度方程残差')
        point_gap-=np.einsum('ij,ij->i',bary,u.reshape((-1,3))[ids,2])
        response.append(dict(relative_free_residual=relative,strain_energy_j=.5*float(u@internal),
            factorization=body.diagnostics['sparse_factorization_ordering']))
    active=forces>eq['certificate']['force_acceptance_n'];mask=active.reshape((n,n))
    # 仅为通用中心P1显示适配使用单patch索引0，不建立齿轮、齿数或传动比。
    grid=RectangularContactGrid(n,n,step,step,origin_xy_m=(float(axis[0]),float(axis[0])))
    geometry=dict(patches=[dict(tooth_offset=0,grid=grid,valid_mask=np.ones((n,n),dtype=bool),first_points_world_m=centers)])
    display_source=dict(converged=True,contact_representation='distributed_P0_surface_cells',halfspace_compliance_used=False,
        selected_tooth_offsets=[0],fields=[dict(pressure_pa=pressure.reshape((n,n)),forces_n=forces.reshape((n,n)),
            valid_mask=np.ones((n,n),dtype=bool),cell_area_m2=area)])
    display=reconstruct_distributed_fe_pressure_field(display_source,geometry=geometry,tooth_offset=0)
    result=dict(normal_force_n=float(forces.sum()),normal_approach_m=float(approach),
        maximum_pressure_pa=float(pressure.max()),raw_P0_contact_area_m2=float(active.sum()*area),
        normal_stiffness_n_per_m=float(np.ones(active.sum())@np.linalg.solve(compliance(np.flatnonzero(active))[active],np.ones(active.sum()))),
        force_equilibrium_relative_error=float((forces.sum()-force_n)/force_n),
        minimum_gauss_remaining_gap_m=float(point_gap.min()),body_response_checks=response,
        contact_window_edge_loaded=bool(mask[0].any() or mask[-1].any() or mask[:,0].any() or mask[:,-1].any()),
        physical_definition='二次初隙的两平面有限弹性块体；底面全固定；不是球体实体或无限体',
        generalized_unit_mapping=dict(reference_length_m=reference_length_m,definition='内部T=F L0、delta=L0 theta；输出仅为物理F和delta'),
        raw_P0=dict(pressure_pa=pressure.reshape((n,n)),forces_n=forces.reshape((n,n)),
            center_points_m=centers,valid_mask=np.ones((n,n),dtype=bool),cell_area_m2=area,spacing_m=step),
        continuous_display=display,quadrature_order=quadrature_order,pressure_filter_applied=False,
        halfspace_compliance_used=False,queried_compliance_columns=len(cache),
        raw_P0_area_definition='force>数值活动门限的投影矩形面积之和',
        display_area_definition='中心C0-P1正压支撑；与原P0面积不同，原中心值和力积分不改变')
    for body in {id(b):b for b in bodies}.values():
        close=getattr(body.factorization,'close',None)
        if close is not None:close()
    return result


def plot_validation(result,reference,output):
    """验证图：原压力与解析参考、剖面、整体量、逐点误差；两语言独立导出。"""
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.tri import Triangulation,LinearTriInterpolator
    from matplotlib import font_manager
    from bevel_gear.publication_style import BLUE,RED,contact_pressure_colormap,figure_style
    plt.rcParams.update(figure_style())
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','Microsoft YaHei','DejaVu Sans'],
        'font.size':8,'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    raw=result['raw_P0'];points=np.asarray(raw['center_points_m']);p=np.asarray(raw['pressure_pa']);n=p.shape[0]
    tris=[]
    for j in range(n-1):
        for i in range(n-1):
            a=j*n+i;tris.extend(((a,a+1,a+n+1),(a,a+n+1,a+n)))
    mesh=Triangulation(points[:,:,0].ravel()*1000,points[:,:,1].ravel()*1000,np.asarray(tris))
    interp=LinearTriInterpolator(mesh,p.ravel());a=reference['contact_radius_m'];p0=reference['peak_pressure_pa']
    q=np.linspace(-1.5*a,1.5*a,501);line=np.asarray(interp(q*1000,np.zeros(len(q))).filled(np.nan))
    analytic=p0*np.sqrt(np.maximum(0.,1-(q/a)**2));r2=points[:,:,0]**2+points[:,:,1]**2
    exact=p0*np.sqrt(np.maximum(0.,1-r2/a**2));error=(p-exact)/p0*100
    for lang in ('zh','en'):
        available={font.name for font in font_manager.fontManager.ttflist}
        choices=('Microsoft YaHei','SimHei','Noto Sans CJK SC','DejaVu Sans') if lang=='zh' else ('Arial','DejaVu Sans')
        plt.rcParams['font.sans-serif']=[next(name for name in choices if name in available)]
        zh=lang=='zh';fig,axes=plt.subplots(2,2,figsize=(7.1,5.35),layout='constrained')
        ax=axes[0,0];upper=max(p.max(),p0)*1e-6
        im=ax.tricontourf(mesh,p.ravel()*1e-6,levels=np.linspace(0,upper,129),cmap=contact_pressure_colormap())
        ax.add_patch(plt.Circle((0,0),a*1000,fill=False,color='white',ls='--',lw=.7))
        ax.set_title('a  FE压力（守恒中心P1）' if zh else 'a  FE pressure (centre-P1)',loc='left',fontsize=9)
        fig.colorbar(im,ax=ax,ticks=np.linspace(0,upper,5),label='接触压力 p (MPa)' if zh else 'Contact pressure p (MPa)')
        ax=axes[0,1];ax.plot(q*1000,analytic*1e-6,color=BLUE,lw=1.4,label='Hertz无限体' if zh else 'Hertz, infinite bodies')
        ax.plot(q*1000,line*1e-6,color=RED,lw=1.1,label='FE中心P1剖面' if zh else 'FE centre-P1 section')
        ax.set(xlabel='投影距离 x (mm)' if zh else 'Projected distance x (mm)',ylabel='接触压力 p (MPa)' if zh else 'Contact pressure p (MPa)')
        ax.set_title('b  中线剖面 y = 0' if zh else 'b  Centre section y = 0',loc='left',fontsize=9);ax.legend(fontsize=7)
        ax=axes[1,0];values=[result['maximum_pressure_pa']/p0,result['raw_P0_contact_area_m2']/reference['contact_area_m2']]
        ax.bar([0,1],values,color=[BLUE,RED],width=.55);ax.axhline(1.,color='#444444',ls='--',lw=.8)
        ax.set_xticks([0,1],['峰压 pmax' if zh else 'Peak pmax','原P0面积 A' if zh else 'Raw P0 area A'])
        ax.set_ylabel('FE / Hertz参考值' if zh else 'FE / Hertz reference');ax.set_ylim(0,max(1.15,max(values)*1.1))
        for i,value in enumerate(values):ax.text(i,1.075,f'{100*(value-1):+.2f}%',ha='center',fontsize=8)
        ax.set_title('c  峰压与原P0面积误差' if zh else 'c  Peak pressure and raw P0 area',loc='left',fontsize=9)
        ax=axes[1,1];limit=max(float(abs(error).max()),1.)
        im=ax.tricontourf(mesh,error.ravel(),levels=np.linspace(-limit,limit,129),cmap='RdBu_r')
        fig.colorbar(im,ax=ax,label='(pFE − pHertz) / pHertz,max (%)')
        ax.set_title('d  原中心值逐点误差' if zh else 'd  Error at original pressure centres',loc='left',fontsize=9)
        for ax in (axes[0,0],axes[1,1]):
            ax.set_aspect('equal');ax.set_xlabel('投影距离 x (mm)' if zh else 'Projected distance x (mm)')
            ax.set_ylabel('投影距离 y (mm)' if zh else 'Projected distance y (mm)')
        for extension in ('png','svg','pdf'):fig.savefig(output/f'hertz_validation.{lang}.{extension}',dpi=300)
        plt.close(fig)


def run_hertz_case(output,*,force_n=1000.,radius_m=.02,young_moduli_pa=(210e9,210e9),poisson_ratios=(.3,.3),
        mesh_size_m=50e-6,pressure_spacing_m=100e-6,quadrature_order=4,width_m=.02,depth_m=.01,bulk_m=.002,
        half_width_m=.0012,refined_depth_m=.001,maximum_nodes=300000,mesh=None,factorization_factory=None,make_figures=True):
    """参数化可运行入口；可传独立SI网格/真实SPD工厂，默认冷启动Gmsh与SciPy。"""
    import numpy as np
    from bevel_gear.elasticity import ElasticMaterial
    from bevel_gear.contact_mesh import boundary_facets
    from bevel_gear.surface_workflow import surface_json_safe
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    materials=[ElasticMaterial(e,v) for e,v in zip(young_moduli_pa,poisson_ratios)]
    reference=hertz_reference(force_n,radius_m,materials)
    if half_width_m<1.5*reference['contact_radius_m']:raise ValueError('计算窗口小于1.5倍Hertz半径，须扩大窗口')
    if mesh is None:mesh=build_block_mesh(width_m=width_m,depth_m=depth_m,h_m=mesh_size_m,bulk_m=bulk_m,
        half_width_m=half_width_m,refined_depth_m=refined_depth_m,maximum_nodes=maximum_nodes)
    nodes=np.asarray(mesh.nodes_m);facets=boundary_facets(mesh)
    keep=(facets['normals'][:,2]>.99999)&np.all(abs(nodes[facets['triangles']][:,:,2])<1e-10,axis=1)
    top={k:v if k in ('nodes_m','metadata') else v[keep] for k,v in facets.items()}
    resolution=verify_surface_resolution(top,1.2*reference['contact_radius_m'],mesh_size_m)
    def write(value,path):path.write_text(json.dumps(surface_json_safe(value),ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n',encoding='utf-8')
    write(asdict(mesh),output/'mesh.json')
    result=solve_hertz_contact(mesh,materials,force_n=force_n,radius_m=radius_m,pressure_spacing_m=pressure_spacing_m,
        half_width_m=half_width_m,quadrature_order=quadrature_order,maximum_dofs=3*maximum_nodes,factorization_factory=factorization_factory)
    peak_error=result['maximum_pressure_pa']/reference['peak_pressure_pa']-1
    area_error=result['raw_P0_contact_area_m2']/reference['contact_area_m2']-1
    result['reference']=reference;result['actual_surface_resolution']=resolution
    result['verification']=dict(peak_relative_error=peak_error,raw_P0_area_relative_error=area_error,
        peak_tolerance=.05,raw_area_tolerance=.05,global_quantities_passed=abs(peak_error)<=.05 and abs(area_error)<=.05,
        force_conservation_passed=abs(result['force_equilibrium_relative_error'])<=1e-9,
        pointwise_pressure_convergence_certified=False,volume_mesh_convergence_certified=False,
        finite_body_approach_difference_from_infinite_reference=result['normal_approach_m']/reference['infinite_body_approach_m']-1,
        interpretation='峰/面积验收不代表逐点压力或体网格已收敛；有限底面约束改变接近量，不按无限体接近量判数值失败')
    write(dict(materials=[asdict(m) for m in materials],force_n=force_n,radius_m=radius_m,
        mesh_size_m=mesh_size_m,pressure_spacing_m=pressure_spacing_m,quadrature_order=quadrature_order,
        mesh_sha256=hashlib.sha256((output/'mesh.json').read_bytes()).hexdigest(),
        block_dimensions_m=[width_m,width_m,depth_m],refined_depth_m=refined_depth_m,maximum_nodes=maximum_nodes,
        figure_contract=dict(conclusion='分布FE的峰/面积与Hertz基准接近，逐点和有限边界差异另列',
            archetype='quantitative_grid',backend='Python_matplotlib',export=['PNG','SVG','PDF'],
            integrity='原P0压力未平滑；中心P1保留原值和力积分；白色虚线仅解析接触圆参考')),
        output/'inputs.json')
    write(result,output/'result.json')
    if make_figures:plot_validation(result,reference,output)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=Path('outputs/hertz_contact'))
    parser.add_argument('--force-n',type=float,default=1000.);parser.add_argument('--radius-m',type=float,default=.02)
    parser.add_argument('--young-pa',type=float,nargs=2,default=[210e9,210e9]);parser.add_argument('--poisson',type=float,nargs=2,default=[.3,.3])
    parser.add_argument('--mesh-um',type=float,default=50.);parser.add_argument('--pressure-um',type=float,default=100.)
    parser.add_argument('--quadrature-order',type=int,default=4);parser.add_argument('--width-m',type=float,default=.02)
    parser.add_argument('--depth-m',type=float,default=.01);parser.add_argument('--bulk-m',type=float,default=.002)
    parser.add_argument('--refined-depth-m',type=float,default=.001)
    parser.add_argument('--half-width-m',type=float,default=.0012);parser.add_argument('--maximum-nodes',type=int,default=300000)
    parser.add_argument('--no-figures',action='store_true');args=parser.parse_args()
    result=run_hertz_case(args.out,force_n=args.force_n,radius_m=args.radius_m,young_moduli_pa=args.young_pa,poisson_ratios=args.poisson,
        mesh_size_m=args.mesh_um*1e-6,pressure_spacing_m=args.pressure_um*1e-6,quadrature_order=args.quadrature_order,
        width_m=args.width_m,depth_m=args.depth_m,bulk_m=args.bulk_m,half_width_m=args.half_width_m,refined_depth_m=args.refined_depth_m,
        maximum_nodes=args.maximum_nodes,make_figures=not args.no_figures)
    print(json.dumps(result['verification'],ensure_ascii=False,indent=2))


if __name__=='__main__':main()
