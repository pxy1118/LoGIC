"""COMSOL primitives shared by analytic verification and actual STL cases."""
import numpy as np
import time
from .common import save, state


def start():
    import mph
    mph.option('session', 'stand-alone')
    return mph.start(cores=8, version='6.3')


def strings(values):
    from jpype import JArray, JString
    return JArray(JString)(list(values))


def plane_selection(comp, tag, z_m, xy_bounds_m,tol=1e-9):
    from jpype import JInt
    comp.selection().create(tag, 'Box')
    s=comp.selection(tag)
    s.set('entitydim',JInt(2))
    s.set('condition','inside')
    lo,hi=xy_bounds_m
    for axis in ['x','y']:
        s.set(axis+'min',str(lo-tol));s.set(axis+'max',str(hi+tol))
    s.set('zmin',str(z_m-tol));s.set('zmax',str(z_m+tol))
    ids=list(map(int,s.entities(2)))
    if not ids:
        raise RuntimeError('Empty boundary selection '+tag)
    return ids


def planar_boundaries(comp, lo, hi, buffer):
    """Select sleeve/port/end-cap planes by coordinates, not entity numbers."""
    from jpype import JInt
    ids=set()
    for axis,values in [('x',[lo,hi]),('y',[lo,hi]),('z',[-buffer,lo,hi,hi+buffer])]:
        for k,value in enumerate(values):
            tag='plane_'+axis+str(k)
            comp.selection().create(tag,'Box');s=comp.selection(tag)
            s.set('entitydim',JInt(2));s.set('condition','inside')
            for name in 'xyz':
                s.set(name+'min',str(-buffer-1e-9));s.set(name+'max',str(hi+buffer+1e-9))
            s.set(axis+'min',str(value-1e-9));s.set(axis+'max',str(value+1e-9))
            ids.update(map(int,s.entities(2)))
    if not ids:raise RuntimeError('No planar mesh boundaries were found')
    return sorted(ids)


def physics(j, inlet='inlet', outlet='outlet', profile='U0', order=1,
            flow_type='CreepingFlow',velocity='1e-4[m/s]',normal_flow=True):
    from jpype import JInt
    c=j.component('comp1')
    j.param().set('rho_f','1000[kg/m^3]')
    j.param().set('mu_f','1e-3[Pa*s]')
    j.param().set('U0',velocity)
    c.material().create('mat1','Common')
    c.material('mat1').propertyGroup('def').set('density',strings(['rho_f']))
    c.material('mat1').propertyGroup('def').set('dynamicviscosity',strings(['mu_f']))
    c.physics().create('spf',flow_type,'geom1')
    f=c.physics('spf')
    f.prop('ShapeProperty').set('order_fluid',JInt(order))
    f.create('inl1','InletBoundary',2)
    f.feature('inl1').selection().named(inlet)
    f.feature('inl1').set('U0in',profile)
    f.create('out1','OutletBoundary',2)
    f.feature('out1').selection().named(outlet)
    f.feature('out1').set('NormalFlow',normal_flow)
    j.study().create('std1')
    j.study('std1').create('stat','Stationary')
    j.study('std1').feature('stat').setSolveFor('/physics/spf',True)
    j.study('std1').setGenPlots(False)


def integral(j, tag, expressions, selection=None, dataset=None):
    j.result().numerical().create(tag,'IntSurface')
    n=j.result().numerical(tag)
    if selection:n.selection().named(selection)
    else:n.selection().all()
    if dataset:
        n.set('data',dataset)
    n.set('expr',strings(expressions))
    values=np.asarray(n.getReal(),dtype=float)
    if values.size!=len(expressions) or not np.isfinite(values).all():
        raise RuntimeError('Invalid integration result '+str(values))
    return values.reshape(-1)


def pressure_section(j, tag, z_m):
    j.result().dataset().create(tag,'CutPlane')
    d=j.result().dataset(tag)
    d.set('quickplane','xy')
    d.set('quickz',str(z_m)+'[m]')
    v=integral(j,'int_'+tag,['p','1'],dataset=tag)
    if v[1]<=0:raise RuntimeError('Nonpositive section area')
    return {'pressure_Pa':float(v[0]/v[1]),'area_m2':float(v[1])}


def solve(j, folder):
    j.study('std1').createAutoSequences('all')
    tags=list(map(str,j.sol().tags()))
    if len(tags)!=1:raise RuntimeError('Unexpected solver sequence')
    solver=j.sol(tags[0])
    solver.feature('s1').set('stol','1e-6')
    state(folder,'SOLVING',solver=tags[0])
    started=time.time()
    solver.runAll()
    return time.time()-started


def solve_sample(case, hmax_mm=0.15, surface_remesh=False,repair_tol_m=1e-9):
    """A development smoke case; not a frozen production label."""
    import pyvista as pv
    from jpype import JDouble, JClass
    from .common import read,digest,now
    from .labels import permeability
    receipt=read(case/'fluid_report.json')
    if digest(case/'fluid_domain.stl')!=receipt['sha256']:
        raise RuntimeError('Fluid boundary changed after validation')
    cfg=read(case/'fluid_config.json');b=cfg['buffer_each_end_mm']*1e-3;L=.0075
    surface=pv.read(case/'fluid_domain.stl')
    surface.points*=1e-3
    surface.save(case/'fluid_domain_m.stl')
    client=start()
    JClass('com.comsol.model.util.ModelUtil').showProgress(str(case/'solver_progress.log'))
    m=client.create('CFD_sample_'+str(read(case/'source.json')['sample_id']));j=m.java
    j.component().create('comp1',True);c=j.component('comp1')
    c.geom().create('geom1',3);g=c.geom('geom1');g.lengthUnit('m');g.run()
    c.mesh().create('mesh1');mesh=c.mesh('mesh1');mesh.contribute('geom/detail',True)
    mesh.create('imp1','Import');imp=mesh.feature('imp1')
    imp.set('filename',str(case/'fluid_domain_m.stl'))
    imp.set('createdom',True);imp.set('facepartition','detectfaces')
    imp.set('stltoltype','absolute');imp.set('stltolabs',str(repair_tol_m))
    imp.set('facemaxangle',JDouble(62))
    state(case,'CFD_IMPORTING',hmax_mm=hmax_mm)
    print('IMPORTING',now(),flush=True)
    imp.importData();mesh.run('imp1')
    m.save(case/'fluid_imported.mph')
    def local_size(feature):
        feature.create('size1','Size');size=feature.feature('size1')
        size.set('custom',True)
        size.set('hmaxactive',True);size.set('hminactive',True)
        size.set('hmax',str(hmax_mm*1e-3));size.set('hmin',str(hmax_mm*1e-4))
        size.set('hgrad',JDouble(1.3));size.set('hcurve',JDouble(0.2));size.set('hnarrow',JDouble(0.5))
    if surface_remesh:
        from jpype import JArray,JInt
        planar=planar_boundaries(c,0,L,b)
        save(case/'planar_remesh_selection.json',{'boundaries':planar,'scope':'coordinate-classified planes only'})
        mesh.create('ftri1','FreeTri');mesh.feature('ftri1').selection().set(JArray(JInt)(planar))
        local_size(mesh.feature('ftri1'))
        print('PLANAR_SURFACE_MESH',len(planar),now(),flush=True);state(case,'CFD_PLANAR_MESHING')
        mesh.run('ftri1')
    mesh.create('ftet1','FreeTet')
    local_size(mesh.feature('ftet1'))
    print('VOLUME_MESH',now(),flush=True);state(case,'CFD_VOLUME_MESHING')
    mesh.run('ftet1')
    stats=mesh.stat()
    from .mesh_quality import volume_mesh_report
    mesh_report=volume_mesh_report(stats,receipt['volume_mm3']*1e-9,L,
                                  'planar' if surface_remesh else 'none')
    # Preserve this costly stage even if a later acceptance check fails.
    m.save(case/'fluid_meshed.mph')
    save(case/'cfd_mesh_report.json',mesh_report)
    if not mesh_report['accepted_for_development_solve']:
        raise RuntimeError('Invalid CFD volume mesh or wrong fluid domain volume')
    selections={'inlet':plane_selection(c,'inlet',-b,(0,L)),
                'outlet':plane_selection(c,'outlet',L+b,(0,L))}
    save(case/'selections.json',selections)
    physics(j,order=1)
    m.save(case/'fluid_presolution.mph')
    state(case,'MESH_READY',hmax_mm=hmax_mm)
    print('SOLVING',now(),flush=True)
    elapsed=solve(j,case)
    m.save(case/'fluid_solution.mph')
    state(case,'SOLVED',elapsed_s=elapsed)
    qi=float(integral(j,'qin',['u*nx+v*ny+w*nz'],selection='inlet')[0])
    qo=float(integral(j,'qout',['u*nx+v*ny+w*nz'],selection='outlet')[0])
    eps=1e-7
    pin=pressure_section(j,'sample_in',eps)
    pout=pressure_section(j,'sample_out',L-eps)
    pi_box=integral(j,'pin_box',['p','1'],selection='inlet')
    po_box=integral(j,'pout_box',['p','1'],selection='outlet')
    result=permeability(qi,qo,pin['pressure_Pa'],pout['pressure_Pa'],L*L,L-2*eps)
    result.update(sample_id=read(case/'source.json')['sample_id'],hmax_mm=hmax_mm,surface_remeshed=surface_remesh,
                  surface_remesh_mode='planar' if surface_remesh else 'none',mesh=mesh_report,
                  repair_tolerance_m=repair_tol_m,pressure_section_epsilon_m=eps,
                  nominal_sample_length_m=L,mesh_quality_review_required=mesh_report['quality_review_required'],
                  delta_p_box_Pa=float(pi_box[0]/pi_box[1]-po_box[0]/po_box[1]),
                  inlet_section=pin,outlet_section=pout,solve_elapsed_s=elapsed,
                  inlet_outer_area_m2=float(pi_box[1]),outlet_outer_area_m2=float(po_box[1]),
                  fluid_sha256=receipt['sha256'],protocol_status='DEVELOPMENT_SMOKE_NOT_PRODUCTION',
                  production_valid=False,code_sha256=digest(__file__))
    if abs(pi_box[1]/(L*L)-1)>1e-3 or abs(po_box[1]/(L*L)-1)>1e-3:
        result.update(cfd_valid=False,logK_valid=False,area_check_failed=True)
    save(case/'results.json',result)
    state(case,'SMOKE_SOLVED' if result['cfd_valid'] else 'QA_FAILED',**result)
    print(result,flush=True)
    client.remove(m)
    return result


def benchmark(folder,hmax=0.0001):
    import math
    folder.mkdir(parents=True,exist_ok=True)
    client=start();m=client.create('CFD_pipe_benchmark');j=m.java
    j.component().create('comp1',True)
    c=j.component('comp1');c.geom().create('geom1',3)
    g=c.geom('geom1');g.lengthUnit('m')
    R=0.0005;L=0.0075
    j.param().set('R',str(R)+'[m]')
    g.create('cyl1','Cylinder');g.feature('cyl1').set('r','R');g.feature('cyl1').set('h',str(L))
    g.run()
    selections={'inlet':plane_selection(c,'inlet',0,(-R,R)),
                'outlet':plane_selection(c,'outlet',L,(-R,R))}
    save(folder/'selections.json',selections)
    c.mesh().create('mesh1');mesh=c.mesh('mesh1');mesh.create('size1','Size')
    size=mesh.feature('size1')
    size.set('custom',True);size.set('hmax',str(hmax));size.set('hmin',str(hmax/10))
    mesh.create('ftet1','FreeTet')
    mesh.run()
    physics(j,profile='2*U0*(1-(x^2+y^2)/R^2)',order=2)
    state(folder,'MESH_READY')
    elapsed=solve(j,folder)
    qi=integral(j,'qin',['u*nx+v*ny+w*nz'],selection='inlet')[0]
    qo=integral(j,'qout',['u*nx+v*ny+w*nz'],selection='outlet')[0]
    pi=pressure_section(j,'pin',0.001)
    po=pressure_section(j,'pout',0.0065)
    dp=pi['pressure_Pa']-po['pressure_Pa']
    area=math.pi*R**2;U=qo/area
    K=0.001*U*0.0055/dp;expected=R**2/8
    error=abs(K/expected-1)
    result={'benchmark_name':'fully developed circular pipe','hmax_m':hmax,'radius_m':R,'measurement_length_m':0.0055,
            'Q_in_m3_s':float(qi),'Q_out_m3_s':float(qo),'delta_p_Pa':dp,
            'K_m2':K,'analytic_K_m2':expected,'K_relative_error':error,
            'mass_imbalance_rel':abs(qi+qo)/abs(qo),'solve_elapsed_s':elapsed,
            'inlet_section':pi,'outlet_section':po,
            'passed':bool(error<0.02 and abs(qi+qo)/abs(qo)<0.001)}
    m.save(folder/'pipe_solution.mph')
    save(folder/'results.json',result)
    state(folder,'QA_PASSED' if result['passed'] else 'QA_FAILED',**result)
    print(result,flush=True)
    if not result['passed']:raise RuntimeError('Analytic pipe verification failed')
    client.remove(m)
    return result
