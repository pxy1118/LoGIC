"""Reuse the actual archived MPH, replacing geometry and dimensional globals."""
import math
import os
import time
import numpy as np
from .common import read,save,state,digest,now
from .comsol import start,integral,plane_selection,physics,solve,strings
from .mesh_quality import volume_mesh_report
from .labels import permeability


def solve_legacy(case):
    import pyvista as pv
    from jpype import JDouble,JClass,JArray,JInt
    cfg=read(case/'protocol.json');gcfg=cfg['geometry'];mcfg=cfg['mesh']
    surface_scope=mcfg.get('surface_remesh_scope','planar')
    if surface_scope not in ('planar','all'):
        raise ValueError('Unsupported COMSOL surface remesh scope')
    template=os.environ.get('LOGIC_COMSOL_TEMPLATE', cfg['provenance']['legacy_template'])
    if digest(template)!=cfg['provenance']['legacy_template_sha256']:
        raise RuntimeError('Archived template hash changed')
    geometry=read(case/'fluid_report.json')
    if digest(case/'fluid_domain.stl')!=geometry['sha256']:
        raise RuntimeError('Fluid surface changed after validation')
    bounds=np.asarray(geometry['bounds_mm'],dtype=float)*.001
    area=(bounds[1]-bounds[0])*(bounds[3]-bounds[2])
    length=bounds[5]-bounds[4]
    if abs(area/cfg['measurement']['reference_area_m2']-1)>.001 or abs(length/cfg['measurement']['measurement_length_m']-1)>.001:
        raise RuntimeError('Observed flow domain differs from the intended dimensions')
    client=start()
    JClass('com.comsol.model.util.ModelUtil').showProgress(str(case/'solver_progress.log'))
    print('LOAD_LEGACY_TEMPLATE',now(),flush=True);state(case,'LOADING_TEMPLATE')
    model=client.load(template);j=model.java;c=j.component('comp1')
    # Preserve the inspected millimetre STL coordinates. Rewriting a binary
    # STL in metres can round nearly coincident cut-face vertices differently.
    g=c.geom('geom1');g.scaleUnitValue(False);g.lengthUnit('mm')
    j.param().set('width',str(math.sqrt(area))+'[m]')
    j.param().set('length',str(length)+'[m]')
    for tag in list(map(str,j.sol().tags())):j.sol(tag).clearSolution()
    # Retain the original material, physics and solver objects, but recreate
    # mesh operations: saved mesh selections/cache are tied to the old STL.
    c.mesh().remove('mesh1');g.run()
    c.mesh().create('mesh1');mesh=c.mesh('mesh1')
    mesh.geometricModel('')
    mesh.contribute('geom/detail',True)
    c.geometricModel('mesh/mesh1')
    mesh.create('imp1','Import');imp=mesh.feature('imp1')
    imp.set('filename',str(case/'fluid_domain.stl'))
    imp.set('createdom',True);imp.set('facepartition','detectfaces');imp.set('facemaxangle',JDouble(62))
    import_tolerance=cfg['mesh'].get('import_tolerance_m',1e-10)
    imp.set('stltoltype','absolute');imp.set('stltolabs',str(import_tolerance*1000))
    print('IMPORTING',now(),flush=True);state(case,'CFD_IMPORTING')
    imp.importData();mesh.run('imp1')
    model.save(case/'fluid_imported.mph')
    bgeom=np.asarray(geometry['bounds_mm'],dtype=float)
    Lgeom=gcfg['specimen_size_mm'];tol=1e-6
    def surface_entities_on_plane(axis,value):
        xyz=np.asarray(mesh.getVertex(),dtype=float).T
        tri=np.asarray(mesh.getElem('tri'),dtype=np.int64).T[:,:3]
        entities=np.asarray(mesh.getElemEntity('tri'),dtype=np.int64)
        if tri.min()<0 or tri.max()>=len(xyz):
            raise RuntimeError('Unexpected mesh vertex indexing')
        counts=np.bincount(entities)
        mask=np.max(np.abs(xyz[tri,axis]-value),axis=1)<=tol
        on_plane=np.bincount(entities[mask],minlength=len(counts))
        return [int(i) for i in np.flatnonzero((counts>0)&(on_plane==counts)) if i>0]
    def local_size(feature):
        feature.create('size1','Size');s=feature.feature('size1')
        s.set('custom',True);s.set('hmaxactive',True);s.set('hminactive',True)
        s.set('hmax',str(mcfg['volume_hmax_mm'])+'[mm]')
        s.set('hmin',str(mcfg['volume_hmin_mm'])+'[mm]')
        for name in ['hgrad','hcurve','hnarrow']:s.set(name,JDouble(mcfg[name]))
    planar=set()
    for axis in range(3):
        values=[float(bgeom[2*axis]),float(bgeom[2*axis+1])]
        if axis==2:values.extend([0,Lgeom])
        for i,value in enumerate(values):
            planar.update(surface_entities_on_plane(axis,value))
    save(case/'planar_remesh_selection.json',{'boundaries':'all' if surface_scope=='all' else sorted(planar),
         'planar_candidates':sorted(planar),'scope':'all faces' if surface_scope=='all' else 'planar faces only'})
    if planar or surface_scope=='all':
        mesh.create('ftri1','FreeTri');feature=mesh.feature('ftri1')
        if surface_scope=='all':feature.selection().all()
        else:feature.selection().set(JArray(JInt)(sorted(planar)))
        local_size(feature)
        print('SURFACE_MESH',surface_scope,len(planar),now(),flush=True)
        state(case,'CFD_SURFACE_MESHING' if surface_scope=='all' else 'CFD_PLANAR_MESHING');mesh.run('ftri1')
    if int(mesh.infoCurrent().getNDomains())==0:
        mesh.create('fresh_domains','CreateDomains');mesh.run('fresh_domains')
    # Mesh features consume the CURRENT mesh numbering. Finalized component
    # numbering can differ (e.g. a tiny closed pocket is numbered first).
    model.save(case/'fluid_surface_ready.mph')
    info=mesh.infoCurrent()
    domains=int(info.getNDomains())
    save(case/'import_report.json',{'geometry_length_unit':str(g.lengthUnit()),
         'domains_after_import':domains,'boundaries_after_import':int(info.getNBoundaries()),
         'component_geometric_model':str(c.geometricModel()),
         'mesh_geometric_model':str(mesh.geometricModel()),
         'absolute_import_tolerance_m':import_tolerance,
         'fluid_sha256':geometry['sha256']})
    print('IMPORTED_DOMAINS',domains,flush=True)
    save(case/'template_compatibility.json',{'recreated_mesh_operations':True,
         'reason':'New STL requires new geometry ownership and selections',
         'old_physics_material_and_solver_retained':True})
    L=gcfg['specimen_size_mm']*.001
    inlet=surface_entities_on_plane(2,float(bgeom[4]))
    outlet=surface_entities_on_plane(2,float(bgeom[5]))
    def adjacent_domains(boundaries):
        return {int(d) for bid in boundaries for d in info.getAdj(JInt(2),JInt(3),JInt(bid)) if int(d)>0}
    inlet_domains=adjacent_domains(inlet);outlet_domains=adjacent_domains(outlet)
    active=sorted(inlet_domains & outlet_domains)
    save(case/'mesh_input_selections.json',{'inlet':inlet,'outlet':outlet,
         'inlet_adjacent_domains':sorted(inlet_domains),'outlet_adjacent_domains':sorted(outlet_domains),
         'active_flow_domains':active,'geometry_length_unit':str(g.lengthUnit())})
    if not active:raise RuntimeError('No connected domain joins the inlet and outlet')
    active_java=JArray(JInt)(active)
    print('ACTIVE_FLOW_DOMAINS',active,flush=True)
    model.save(case/'fluid_imported.mph')
    mesh.create('ftet1','FreeTet');feature=mesh.feature('ftet1')
    feature.selection().set(active_java);local_size(feature)
    print('VOLUME_MESH',now(),flush=True);state(case,'CFD_VOLUME_MESHING')
    started=time.time();mesh.run();mesh_elapsed=time.time()-started
    class SIStatistics:
        def __init__(self,statistics):self.statistics=statistics
        def __getattr__(self,name):return getattr(self.statistics,name)
        def getVolume(self,typ):return float(self.statistics.getVolume(typ))*1e-9
        def getMinVolume(self,typ):return float(self.statistics.getMinVolume(typ))*1e-9
    report=volume_mesh_report(SIStatistics(mesh.stat()),geometry['volume_mm3']*1e-9,
                              gcfg['specimen_size_mm']*.001,
                              ('external_3matic_'+'_'.join(gcfg['surface_remesh_sequence'])+'_then_COMSOL_planar')
                              if gcfg['surface_remesh_sequence'] else 'COMSOL_planar_faces_only')
    if surface_scope=='all':report['surface_remesh_mode']='COMSOL_all_faces_after_'+gcfg['boolean_backend']
    report['elapsed_s']=mesh_elapsed
    model.save(case/'fluid_meshed.mph');save(case/'cfd_mesh_report.json',report)
    if not report['accepted_for_development_solve']:
        raise RuntimeError('Invalid volume mesh, units or volume closure')
    return _solve_flow_model(client,model,case,cfg,geometry,report,active,import_tolerance)


def _solve_flow_model(client,model,case,cfg,geometry,report,active,import_tolerance):
    from jpype import JArray,JInt
    j=model.java;c=j.component('comp1');g=c.geom('geom1')
    gcfg=cfg['geometry'];bgeom=np.asarray(geometry['bounds_mm'],dtype=float)
    bounds=bgeom*.001;area=(bounds[1]-bounds[0])*(bounds[3]-bounds[2])
    length=bounds[5]-bounds[4];L=gcfg['specimen_size_mm']*.001;tol=1e-6
    def adjacent_domains(boundaries):
        return {int(d) for bid in boundaries for d in g.getAdj(JInt(2),JInt(3),JInt(bid)) if int(d)>0}
    lo=float(min(bgeom[0],bgeom[2]));hi=float(max(bgeom[1],bgeom[3]))
    for name in ['aligned_inlet','aligned_outlet']:
        if name in list(map(str,c.selection().tags())):c.selection().remove(name)
    inlet=plane_selection(c,'aligned_inlet',float(bgeom[4]),(lo,hi),tol=tol)
    outlet=plane_selection(c,'aligned_outlet',float(bgeom[5]),(lo,hi),tol=tol)
    info=g
    final_in=adjacent_domains(inlet);final_out=adjacent_domains(outlet)
    final_active=sorted(final_in & final_out)
    if not final_active:raise RuntimeError('Final mesh lost inlet/outlet connectivity')
    save(case/'selections.json',{'inlet':inlet,'outlet':outlet,'active_flow_domains':final_active,
         'input_mesh_active_flow_domains':active,'geometry_length_unit':str(g.lengthUnit())})
    active_java=JArray(JInt)(final_active)
    # Reinstantiate the same physical equations and constants on the current
    # mesh. Clearing old solution values alone leaves stale equation caches.
    for tag in list(map(str,j.sol().tags())):j.sol().remove(tag)
    for tag in list(map(str,j.study().tags())):j.study().remove(tag)
    c.physics().remove('spf')
    for tag in list(map(str,c.material().tags())):c.material().remove(tag)
    physics(j,inlet='aligned_inlet',outlet='aligned_outlet',order=1,
            flow_type='LaminarFlow',velocity=str(cfg['physics']['inlet_velocity_m_s'])+'[m/s]',
            normal_flow=False)
    f=c.physics('spf');f.selection().set(active_java)
    f.prop('PhysicalModelProperty').set('Compressibility','Incompressible')
    f.prop('PhysicalModelProperty').set('StokesFlowProp',False)
    f.feature('out1').set('SuppressBackflow',True)
    f.prop('PhysicalModelProperty').set('IncludeGravity',False)
    c.cpl('aveop1').selection().named('aligned_inlet')
    c.cpl('aveop2').selection().named('aligned_outlet')
    c.cpl('intop1').selection().set(active_java)
    for tag in list(map(str,c.material().tags())):c.material(tag).selection().set(active_java)
    j.study('std1').setGenPlots(False)
    model.save(case/'fluid_presolution.mph')
    print('SOLVING_NAVIER_STOKES',now(),flush=True);state(case,'SOLVING_NAVIER_STOKES')
    elapsed=solve(j,case)
    solver_tag=str(list(j.sol().tags())[0])
    solution_size=list(map(int,j.sol(solver_tag).getSize()))
    if not solution_size or max(solution_size)<=1:
        raise RuntimeError('Solver produced no effective degrees of freedom')
    model.save(case/'fluid_solution.mph')
    data='aligned_solution'
    if data in list(map(str,j.result().dataset().tags())):j.result().dataset().remove(data)
    j.result().dataset().create(data,'Solution');j.result().dataset(data).set('solution',solver_tag)
    qi=float(integral(j,'aligned_qin',['u*nx+v*ny+w*nz'],selection='aligned_inlet',dataset=data)[0])
    qo=float(integral(j,'aligned_qout',['u*nx+v*ny+w*nz'],selection='aligned_outlet',dataset=data)[0])
    pin=integral(j,'aligned_pin',['p','1'],selection='aligned_inlet',dataset=data)
    pout=integral(j,'aligned_pout',['p','1'],selection='aligned_outlet',dataset=data)
    if max(abs(pin[1]/area-1),abs(pout[1]/area-1))>.001:
        raise RuntimeError('Inlet/outlet area differs from configured SI reference area')
    result=permeability(qi,qo,float(pin[0]/pin[1]),float(pout[0]/pout[1]),area,length)
    # Rename specimen-specific fields: this protocol measures the full flow domain.
    result['K_domain_app_z_m2']=result.pop('K_app_z_m2')
    result['log10_K_domain_app_z']=result.pop('log10_K_app_z')
    result['ln_K_domain_app_z']=math.log(result['K_domain_app_z_m2'])
    result['delta_p_domain_Pa']=result.pop('delta_p_sample_Pa')
    result['measurement_length_m']=result.pop('sample_length_m')
    j.result().numerical().create('aligned_kappa','EvalGlobal')
    kappa_eval=j.result().numerical('aligned_kappa');kappa_eval.set('expr',strings(['kappa']));kappa_eval.set('data',data)
    result.update(sample_id=read(case/'source.json')['sample_id'],mesh=report,
                  inlet_average_pressure_Pa=float(pin[0]/pin[1]),
                  outlet_average_pressure_Pa=float(pout[0]/pout[1]),
                  inlet_area_m2=float(pin[1]),outlet_area_m2=float(pout[1]),
                  solve_elapsed_s=elapsed,template_kappa_m2=float(np.asarray(kappa_eval.getReal()).item()),
                  solver_sequence=solver_tag,solution_size=solution_size,solution_dataset=data,
                  nominal_specimen_length_m=L,geometry_length_unit=str(g.lengthUnit()),
                  physics='steady incompressible Navier-Stokes',inlet_velocity_m_s=.001,
                  production_valid=False,protocol_status='LEGACY_ALIGNED_PILOT_NOT_PRODUCTION',
                  mesh_quality_review_required=report['quality_review_required'],
                  fluid_sha256=geometry['sha256'],protocol_sha256=digest(case/'protocol.json'))
    result.update(observed_domain_bbox_mm=geometry['bounds_mm'],
                  intended_reference_area_m2=cfg['measurement']['reference_area_m2'],
                  intended_measurement_length_m=cfg['measurement']['measurement_length_m'],
                  geometry_preprocessing_backend=gcfg['boolean_backend'],
                  geometry_surface_remesh_sequence=gcfg['surface_remesh_sequence'],
                  geometry_surface_export_format=gcfg.get('surface_export_format','binary_float32'))
    result['import_tolerance_m']=import_tolerance
    result['template_kappa_relative_difference']=abs(result['template_kappa_m2']/result['K_domain_app_z_m2']-1)
    if result['template_kappa_relative_difference']>.001:
        result.update(cfd_valid=False,logK_valid=False,template_consistency_failed=True)
    save(case/'results.json',result)
    state(case,'ALIGNED_SMOKE_SOLVED' if result['cfd_valid'] else 'QA_FAILED',**result)
    print(result,flush=True);client.remove(model)
    return result


def solve_from_native_mesh(case,source):
    """Rebuild equations and solver while retaining a validated cached mesh."""
    from jpype import JClass
    cfg=read(case/'protocol.json');geometry=read(case/'fluid_report.json')
    mesh_path=source/'fluid_meshed.mph';report=read(source/'cfd_mesh_report.json')
    if not report['accepted_for_development_solve']:raise RuntimeError('Cached mesh was not accepted')
    if digest(source/'fluid_domain.stl')!=geometry['sha256']:
        raise RuntimeError('Cached mesh geometry differs')
    save(case/'mesh_cache_origin.json',{'path':str(mesh_path),'sha256':digest(mesh_path),
         'source_protocol_sha256':digest(source/'protocol.json'),
         'source_mesh_report_sha256':digest(source/'cfd_mesh_report.json')})
    client=start();JClass('com.comsol.model.util.ModelUtil').showProgress(str(case/'solver_progress.log'))
    state(case,'LOADING_VERIFIED_MESH');print('LOADING_VERIFIED_MESH',now(),flush=True)
    model=client.load(mesh_path);mesh=model.java.component('comp1').mesh('mesh1')
    if int(mesh.stat().getNumElem('tet'))!=report['elements']:
        raise RuntimeError('Cached native mesh element count changed')
    save(case/'cfd_mesh_report.json',report)
    active=read(source/'mesh_input_selections.json')['active_flow_domains']
    return _solve_flow_model(client,model,case,cfg,geometry,report,active,
                             cfg['mesh'].get('import_tolerance_m',1e-10))
