from __future__ import annotations
import importlib
import shutil
import sys
import platform
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.stats import qmc
from .common import ROOT, PROJECT, read, save, digest, object_hash, safe_id, state, now
from .component_cleanup import cleanup_components,triangle_surface
from .storage import verify_case_files


def create_run(run_id, cfg, pilot=False):
    folder = ROOT / 'runs' / safe_id(run_id)
    if folder.exists():
        raise FileExistsError('Run already exists; use its frozen configuration and resume')
    source = folder / 'protocol' / 'source'
    source.mkdir(parents=True)
    hashes = {}
    for name in ('tpms_hybrid.py', 'orientation_mapping.py'):
        shutil.copy2(PROJECT / name, source / name)
        hashes[name] = digest(source / name)
    pipeline_hashes={}
    for path in [ROOT/'main.py',*sorted((ROOT/'src').glob('*.py')),*sorted((ROOT/'workers').glob('*.py'))]:
        relative=path.relative_to(ROOT); destination=folder/'protocol/pipeline'/relative
        destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,destination)
        pipeline_hashes[relative.as_posix()]=digest(destination)
    save(folder / 'config.json', cfg)
    count = 6 if pilot else cfg['samples']
    lhs = qmc.LatinHypercube(d=32, seed=cfg['seed']).random(count)
    records = []
    for i in range(count):
        design = sample_design(i, lhs[i], cfg, pilot)
        design['design_hash'] = object_hash(design)
        case = folder / 'samples' / design['sample_id']
        save(case / 'design.json', design)
        state(case, 'PLANNED')
        records.append(dict(sample_id=design['sample_id'], family=design['family'], kind=design['kind'], design_hash=design['design_hash']))
    save(folder / 'manifest.json', dict(schema_version=1, created_utc=now(), run_id=run_id,
         purpose='calibration' if pilot else 'dataset', config_sha256=digest(folder/'config.json'),
         generator_hashes=hashes, pipeline_hashes=pipeline_hashes,count=count, samples=records, seed=cfg['seed'],
         environment=dict(python=sys.version, platform=platform.platform(), numpy=np.__version__)))
    return folder


def sample_design(i, u, cfg, pilot):
    xyz = np.moveaxis(np.indices((3,3,3)), 0, -1).astype(float) - 1
    weights = np.zeros((3,3,3,3))
    rho_min,rho_max=cfg['geometry']['density_range']
    density = np.full((3,3,3), .30 if pilot else rho_min + (rho_max-rho_min)*u[0])
    angles = np.zeros((3,3,3,3))
    families = [(0,), (1,), (2,), (0,1), (0,2), (1,2), (0,1,2)]
    if pilot:
        family = families[i] if i < 3 else (0,1,2)
        if i < 3:
            weights[...,i] = 1
        else:
            labels = np.indices((3,3,3))[2]
            weights = np.eye(3)[labels]
            if i == 3:
                density = .30 + .04*xyz[...,2]
            elif i == 4:
                angles[...,1] = 10*(xyz[...,2]+1)
            else:
                density = .25 + .03*(xyz[...,0]+1)
                angles[...,1] = 7*(xyz[...,2]+1)
        kind = 'calibration'
    else:
        kind = 'reference' if i < 50 else ('coverage' if i < 450 else 'boundary')
        if i>=cfg['samples']:
            kind='reference' if u[31]<.1 else ('coverage' if u[31]<.9 else 'boundary')
        family = families[i % (3 if kind == 'reference' else 7)]
        if len(family) == 1:
            weights[...,family[0]] = 1
        else:
            # Smooth random score fields; categorical controls, continuous spatial interpolation.
            scores = []
            for k, _ in enumerate(family):
                coeff = 2*u[3+4*k:7+4*k]-1
                scores.append(coeff[0] + np.einsum('...j,j->...', xyz, coeff[1:]))
            labels = np.array(family)[np.argmax(np.stack(scores,-1),-1)]
            # Guarantee each requested topology is represented at a deterministic control point.
            for k, channel in enumerate(family):
                labels[0,0,k] = channel
            weights = np.eye(3)[labels]
        if kind != 'reference':
            grad = (u[16:19]-.5) * (.06 if kind=='boundary' else .035)
            density = np.clip(density + np.einsum('...j,j->...',xyz,grad), rho_min,rho_max)
        base = (u[19:22]-.5)*120
        slopes = (u[22:31].reshape(3,3)-.5) * (10 if kind=='boundary' else 6)
        if kind == 'reference':
            slopes[:] = 0
        angles = base + np.einsum('...j,kj->...k',xyz,slopes)
    quat = Rotation.from_euler('xyz',angles.reshape(-1,3),degrees=True).as_quat().reshape(3,3,3,4)
    max_relative = 0.
    for axis in range(3):
        a=np.take(quat,[0,1],axis=axis); b=np.take(quat,[1,2],axis=axis)
        dots=np.clip(np.abs(np.sum(a*b,axis=-1)),0,1)
        max_relative=max(max_relative,float(np.degrees(2*np.arccos(dots)).max()))
    if max_relative > cfg['geometry']['max_neighbor_rotation_deg']:
        raise ValueError('Sampling exceeded physical neighboring rotation limit')
    return dict(sample_id=('p' if pilot else 's')+f'{i:04d}', kind=kind, family=list(family),
                seed=cfg['seed'], candidate_index=i, weights=weights.tolist(), density=density.tolist(),
                rotation_deg=angles.tolist(), quaternion_xyzw=quat.tolist(), max_neighbor_rotation_deg=max_relative)


def extend_run(run,count,reason):
    cfg=read(run/'config.json');manifest=read(run/'manifest.json')
    if manifest['purpose']!='dataset':raise ValueError('Use a new pilot for calibration changes')
    if count<1 or not reason.strip():raise ValueError('Positive count and an audit reason required')
    start=manifest['count'];before=digest(run/'manifest.json')
    lhs=qmc.LatinHypercube(d=32,seed=np.random.SeedSequence([cfg['seed'],start]).generate_state(1)[0].item()).random(count)
    for i,u in enumerate(lhs,start):
        design=sample_design(i,u,cfg,False);design['design_hash']=object_hash(design)
        case=run/'samples'/design['sample_id'];save(case/'design.json',design);state(case,'PLANNED')
        manifest['samples'].append(dict(sample_id=design['sample_id'],family=design['family'],kind=design['kind'],design_hash=design['design_hash']))
    manifest['count']+=count
    manifest.setdefault('extensions',[]).append(dict(start=start,count=count,reason=reason,utc=now(),previous_manifest_sha256=before))
    save(run/'manifest.json',manifest)
    return dict(candidates=manifest['count'],target_valid=cfg['samples'])


def surface_report(mesh, size):
    mesh = triangle_surface(mesh)
    connected=mesh.connectivity()
    regions=np.asarray(connected.cell_data['RegionId'],dtype=int)
    triangles=connected.faces.reshape(-1,4)[:,1:]
    vertices=np.asarray(connected.points,dtype=np.float64)[triangles]
    signed=np.einsum('ij,ij->i',vertices[:,0],np.cross(vertices[:,1],vertices[:,2]))/6
    volumes=np.sort(np.abs(np.bincount(regions,weights=signed)))[::-1]
    return dict(points=mesh.n_points, triangles=mesh.n_cells, bounds_mm=list(mesh.bounds),
                components=int(np.unique(regions).size), open_edges=mesh.n_open_edges,
                component_volumes_mm3=volumes.tolist(),
                largest_component_volume_fraction=float(volumes[0]/volumes.sum()) if volumes.sum()>0 else None,
                volume_mm3=float(abs(mesh.volume)), relative_density=float(abs(mesh.volume)/size**3))


def crop_cube(mesh,span):
    import pyvista as pv
    import trimesh
    mesh=mesh.triangulate().clean()
    triangles=mesh.faces.reshape(-1,4)[:,1:]
    solid=trimesh.Trimesh(vertices=mesh.points,faces=triangles,process=False)
    if not solid.is_volume:raise ValueError('Uncropped scalar surface is not a consistently oriented closed volume')
    # Robust Boolean avoids vtkClipClosedSurface cap failures at successive cube edges.
    cut=trimesh.boolean.intersection([solid,trimesh.creation.box(extents=[span]*3)],engine='manifold',check_volume=True)
    if not cut.is_volume:raise ValueError('Cube intersection is not a valid volume')
    return pv.PolyData(cut.vertices,np.column_stack([np.full(len(cut.faces),3),cut.faces])).triangulate().clean()


def generate(case, cfg, source):
    import pyvista as pv
    case, source = case.resolve(), source.resolve()
    receipt = case/'geometry_quality.json'
    if receipt.exists():
        existing=read(receipt)
        verify_case_files(case,existing['hashes'])
        if not existing['passed']:
            raise RuntimeError('Previously rejected geometry; preserve record and use another candidate')
        return existing
    for key in ('tpms_hybrid','orientation_mapping'):
        sys.modules.pop(key,None)
    sys.path.insert(0,str(source))
    try:
        gen=importlib.import_module('tpms_hybrid')
        d=read(case/'design.json'); g=cfg['geometry']; n=g['resolution']; size=g['size_mm']
        # Explicitly fix domain scale; never infer scale from the meshed bounding box.
        span=3.*g['periods']; bounds=(-span/2,span/2)
        w=np.array(d['weights']); rho=np.array(d['density']); rot=np.array(d['rotation_deg'])
        sampling_step_mm=size/(n-1)
        # Hold the physical smoothing width fixed in resolution comparisons.
        # Missing length preserves the old 0.5-voxel protocol for frozen runs.
        sigma_mm=g.get('weight_smoothing_sigma_mm',.5*sampling_step_mm)
        sigma_voxels=sigma_mm/sampling_step_mm
        W=gen.generate_weight_matrix(n,bounds,bounds,bounds,w,smooth_sigma=sigma_voxels)
        D=gen.generate_density_field_from_3x3x3(n,bounds,bounds,bounds,rho)
        Q=gen.generate_rotation_field_from_3x3x3(n,bounds,bounds,bounds,rot)
        quality={}
        phase=np.array(g.get('phase_offset_periods',[0,0,0]))*3.
        functions=[(lambda x,y,z,f=f: f(x+phase[0],y+phase[1],z+phase[2]))
                   for f in [getattr(gen,name) for name in g['topologies']]]
        boundary_method=g.get('boundary_method','manifold_boolean')
        boundary_options={'surface_boundary_mode':'implicit_box'} if boundary_method=='implicit_box' else {}
        mesh,porosity,threshold,mask=gen.create_hybrid_tpms_solid(
            functions,bounds,bounds,bounds,W,
            density_field=D,rotation_field=Q,resolution=n,return_mask=True,
            rotation_mode='integrated',rotation_diagnostics=quality,
            max_axis_error_deg=g['max_axis_error_deg'],max_phase_stretch_ratio=g['max_phase_stretch_ratio'],
            morph_close=False,remove_small=False,smooth=False,verbose=False,**boundary_options)
        mesh=mesh.triangulate().clean() if boundary_method=='implicit_box' else crop_cube(mesh,span)
        mesh.points=np.asarray(mesh.points,dtype=np.float64)
        mesh.scale(size/span,inplace=True); mesh.translate((size/2,)*3,inplace=True)
        mesh=triangle_surface(mesh)
        # Collapse Boolean cap needles with a bounded physical tolerance, retaining all components.
        import manifold3d
        before_volume=abs(mesh.volume)
        if g.get('surface_simplify_tolerance_mm',0)>0:
            manifold=manifold3d.Manifold(manifold3d.Mesh(np.asarray(mesh.points,dtype=np.float32),
                             np.asarray(mesh.faces.reshape(-1,4)[:,1:],dtype=np.uint32)))
            simplified=manifold.as_original().simplify(g['surface_simplify_tolerance_mm']).to_mesh()
            mesh=pv.PolyData(simplified.vert_properties[:,:3],
                       np.column_stack([np.full(len(simplified.tri_verts),3),simplified.tri_verts])).clean()
        simplification_drift=float(abs(mesh.volume)/before_volume-1)
        cleanup_cfg=g.get('component_cleanup',{})
        mesh,removed,cleanup=cleanup_components(mesh,enabled=cleanup_cfg.get('enabled',False),
            max_removed_volume_fraction=cleanup_cfg.get('max_removed_volume_fraction',.01))
        if removed is not None:
            removed.save(case/'removed_components.stl')
            cleanup['removed_surface_sha256']=digest(case/'removed_components.stl')
        mesh.save(case/'source.stl')
        cleanup['retained_surface_sha256']=digest(case/'source.stl')
        save(case/'component_cleanup.json',cleanup)
        raw_mask=mask
        if cleanup_cfg.get('enabled',False):
            reference=pv.ImageData(dimensions=(n,n,n),spacing=(sampling_step_mm,)*3,origin=(0,0,0))
            sampled=mesh.voxelize_binary_mask(reference_volume=reference)
            mask=np.asarray(sampled.point_data['mask']).reshape((n,n,n),order='F').astype(bool)
        np.savez_compressed(case/'inputs.npz',weights=w,density=rho,rotation_deg=rot,
                            topology_names=np.asarray(g['topologies']),
                            quaternion_xyzw=np.array(d['quaternion_xyzw']),source_voxel=mask,
                            raw_generator_voxel=raw_mask,source_voxel_origin_mm=np.zeros(3),
                            source_voxel_spacing_mm=np.full(3,sampling_step_mm))
        # Inspect the serialized geometry used by the mesher, including STL vertex merging.
        report=surface_report(pv.read(case/'source.stl'),size)
        failures=[]
        if not cleanup['passed']:failures.append('detached components exceed the small-fragment volume limit')
        if report['components']!=1 or report['open_edges']!=0:
            failures.append('source must be one closed manifold without removing fragments')
        if not np.allclose(report['bounds_mm'],[0,size]*3,atol=cfg['quality']['bounds_tolerance_mm'],rtol=0):
            failures.append('source bounds do not match nominal cube')
        result=dict(passed=not failures,failures=failures,surface=report,rotation=quality,
                    topologies=g['topologies'],
                    resolution=n,sampling_step_mm=sampling_step_mm,
                    weight_smoothing_sigma_mm=sigma_mm,weight_smoothing_sigma_voxels=sigma_voxels,
                    component_cleanup=cleanup,
                    source_voxel_scope='cropped and cleaned STL sampled on nominal grid' if cleanup_cfg.get('enabled',False) else 'raw generator mask',
                    threshold=threshold,source_voxel_porosity=float(1-mask.mean()),generator_voxel_porosity=float(porosity),affine_scale=size/span,
                    phase_offset_periods=g.get('phase_offset_periods',[0,0,0]),
                    crop_method=('continuous scalar intersection with exact box before contouring' if boundary_method=='implicit_box' else 'manifold exact cube intersection') + ('; recorded small-component cleanup' if cleanup_cfg.get('enabled',False) else '; no component removal'),
                    surface_simplify_tolerance_mm=g.get('surface_simplify_tolerance_mm',0),
                    simplification_volume_drift=simplification_drift,
                    hashes={p:digest(case/p) for p in ['source.stl','inputs.npz','design.json','component_cleanup.json']+
                            (['removed_components.stl'] if removed is not None else [])})
        save(receipt,result); state(case,'GEOMETRY_OK' if not failures else 'GEOMETRY_REJECTED')
        if failures: raise RuntimeError('; '.join(failures))
        return result
    finally:
        sys.path.remove(str(source))
