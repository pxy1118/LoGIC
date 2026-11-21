from __future__ import annotations
import numpy as np
from .common import read, save, digest, verify_files, state
from .design import surface_report
from .storage import verify_case_files


def edge_length_statistics(coords,conn,pairs):
    """Measure corner edges; quadratic midside nodes do not halve element size."""
    lengths=np.empty((len(conn),len(pairs)),dtype=np.float64)
    for start in range(0,len(conn),20000):
        stop=min(start+20000,len(conn));points=coords[conn[start:stop]]
        for index,(a,b) in enumerate(pairs):
            lengths[start:stop,index]=np.linalg.norm(points[:,a]-points[:,b],axis=1)
    values=lengths.ravel()
    quantiles=np.quantile(values,[0,.01,.1,.5,.9,.99,1])
    return dict(zip(['min','p01','p10','median','p90','p99','max'],map(float,quantiles)),
        below_0p05_fraction=float(np.mean(values<.05)),below_0p10_fraction=float(np.mean(values<.10)),
        count=len(values),units='mm',scope='Element edge incidences; shared edges counted per element. Corner edges only.')


def read_tet10(path):
    nodes, elements, mode, pending, part_count = {}, [], None, [], 0
    with path.open(encoding='utf-8',errors='replace') as stream:
        for line in stream:
            line=line.strip()
            if not line or line.startswith('**'): continue
            if line.startswith('*'):
                if pending: raise ValueError('Truncated element connectivity')
                upper=line.upper(); mode=None
                if upper.startswith('*PART,'): part_count+=1
                if upper=='*NODE' or upper.startswith('*NODE,'): mode='node'
                elif upper.startswith('*ELEMENT,'):
                    if 'TYPE=C3D10' in upper.replace(' ',''): mode='element'
                    else: raise ValueError('Expected exclusively Tet10 volume elements')
                elif upper.startswith('*INSTANCE,'): mode='instance'
                continue
            values=[x.strip() for x in line.split(',') if x.strip()]
            if mode=='node':
                label=int(values[0])
                if label in nodes: raise ValueError('Duplicate node IDs')
                nodes[label]=[float(v) for v in values[1:4]]
            elif mode=='element':
                pending.extend(int(x) for x in values)
                if len(pending)==11: elements.append(pending); pending=[]
                elif len(pending)>11: raise ValueError('Unexpected element connectivity')
            elif mode=='instance':
                raise ValueError('Non-identity instance transforms require explicit handling')
    if part_count>1 or not nodes or not elements or pending:
        raise ValueError('Expected one complete nonempty Tet10 part')
    labels=np.array(sorted(nodes),dtype=np.int64)
    coords=np.array([nodes[int(k)] for k in labels])
    element=np.asarray(elements,dtype=np.int64)
    if not np.isfinite(coords).all() or len(np.unique(element[:,0]))!=len(element):
        raise ValueError('Nonfinite coordinates or duplicate element IDs')
    conn=np.searchsorted(labels,element[:,1:])
    if np.any(conn>=len(labels)) or not np.array_equal(labels[conn],element[:,1:]):
        raise ValueError('Element references nonexistent nodes')
    return labels,coords,element[:,0],conn


def tet10_derivative(bary):
    L=np.asarray(bary); dL=np.array([[-1,-1,-1],[1,0,0],[0,1,0],[0,0,1]])
    rows=[(4*L[i]-1)*dL[i] for i in range(4)]
    for i,j in [(0,1),(1,2),(2,0),(0,3),(1,3),(2,3)]:
        rows.append(4*(L[i]*dL[j]+L[j]*dL[i]))
    return np.array(rows)


def mesh_quality(coords, conn):
    a,b=.5854101966249685,.1381966011250105
    gauss=[np.roll([a,b,b,b],i) for i in range(4)]
    check=gauss+[np.full(4,.25)]+list(np.eye(4))
    volumes=np.empty(len(conn)); min_dets=np.empty(len(conn)); shapes=np.empty(len(conn))
    for start in range(0,len(conn),20000):
        end=min(start+20000,len(conn)); points=coords[conn[start:end]]
        jacobians=[np.einsum('npi,pj->nij',points,tet10_derivative(L)) for L in check]
        det=np.stack([np.linalg.det(j) for j in jacobians],axis=1)
        volumes[start:end]=det[:,:4].sum(axis=1)/24
        min_dets[start:end]=det.min(axis=1)
        sv=np.linalg.svd(jacobians[4],compute_uv=False)
        shapes[start:end]=sv[:,-1]/np.maximum(sv[:,0],1e-30)
    return dict(nodes=len(coords),elements=len(conn),volume_mm3=float(volumes.sum()),
                min_sampled_jacobian=float(min_dets.min()),nonpositive_jacobian_elements=int((min_dets<=0).sum()),
                min_element_volume_mm3=float(volumes.min()),shape_ratio_min=float(shapes.min()),
                shape_ratio_p01=float(np.quantile(shapes,.01)),
                jacobian_scope='Tet10 derivatives at 4 quadrature, center and 4 vertex points; datacheck still required')


def straight_tet10(coords, conn):
    """Elevate the corner tetrahedra using one exact midpoint per unique edge."""
    corner_ids, corners=np.unique(conn[:,:4],return_inverse=True)
    corners=corners.reshape(-1,4); points=coords[corner_ids]
    jac=np.stack([points[corners[:,i]]-points[corners[:,0]] for i in (1,2,3)],axis=-1)
    if np.any(np.linalg.det(jac)<=0):raise ValueError('Nonpositive corner tetrahedron; cannot elevate')
    pairs=np.array([(0,1),(1,2),(2,0),(0,3),(1,3),(2,3)])
    edges=np.sort(corners[:,pairs].reshape(-1,2),axis=1)
    unique,inverse=np.unique(edges,axis=0,return_inverse=True)
    midpoint=(points[unique[:,0]]+points[unique[:,1]])/2
    elevated=np.column_stack((corners,len(points)+inverse.reshape(-1,6)))
    return np.vstack((points,midpoint)),elevated


def reconstruct_midpoints(case,cfg):
    """Preserve native export; explicitly create straight Tet10 geometry before QA."""
    if cfg['mesh'].get('midside_policy','native')!='exact_edge_midpoints':return
    receipt=case/'midpoint_reconstruction.json'
    if receipt.exists():verify_case_files(case,read(receipt)['hashes']);return
    status=read(case/'mesher_result.json')
    if status['status']!='MESH_EXPORTED' or status.get('midpoint_projection')!='disabled':
        raise ValueError('Exact midpoint reconstruction requires an unprojected native export')
    if (case/'volume.inp').exists():raise ValueError('Unrecorded volume exists; preserve it for inspection')
    raw=case/'volume_3matic.inp'
    labels,coords,eids,conn=read_tet10(raw)
    repeated=int(np.sum(np.any(np.diff(np.sort(conn,axis=1),axis=1)==0,axis=1)))
    output,new_conn=straight_tet10(coords,conn)
    temp=case/'volume.inp.tmp'
    with temp.open('w',encoding='ascii',newline='\n') as f:
        f.write('** Straight Tet10 reconstructed from preserved 3-matic corner geometry\n*Node\n')
        for label,xyz in enumerate(output,1):f.write('%d, %.16g, %.16g, %.16g\n'%(label,*xyz))
        f.write('*Element, type=C3D10\n')
        for eid,row in zip(eids,new_conn+1):f.write(', '.join(map(str,[eid,*row]))+'\n')
    temp.replace(case/'volume.inp')
    save(receipt,dict(method='one exact arithmetic midpoint per unique undirected corner edge',
         geometry='straight tetrahedra; corner coordinates and connectivity unchanged; node labels reassigned',
         native_nodes=len(coords),output_nodes=len(output),elements=len(conn),
         native_elements_with_repeated_node_ids=repeated,
         hashes={name:digest(case/name) for name in ('volume_3matic.inp','volume.inp')}))


def validate_mesh(case,cfg):
    import pyvista as pv
    receipt=case/'mesh_quality.json'
    if receipt.exists():
        data=read(receipt); verify_case_files(case,data['hashes'])
        if not data['passed']: raise RuntimeError('Previously rejected volume mesh')
        if 'voxel_fea.npz' in data['hashes'] and 'geometry_features.json' in data['hashes']:return data
    status=read(case/'mesher_result.json')
    if status['status']!='MESH_EXPORTED':
        state(case,'MESH_REJECTED',reason='3-matic meshing failed',
              self_intersections=status.get('self_intersections'),
              overlapping_triangles=status.get('overlapping_triangles'))
        raise RuntimeError(status.get('error','3-matic meshing failed'))
    reconstruct_midpoints(case,cfg)
    size=cfg['geometry']['size_mm']; qcfg=cfg['quality']
    surface=pv.read(case/'remeshed.stl').triangulate().clean()
    report=surface_report(surface,size)
    report['edge_lengths']=edge_length_statistics(np.asarray(surface.points),
        surface.faces.reshape(-1,4)[:,1:],[(0,1),(1,2),(2,0)])
    labels,coords,eids,conn=read_tet10(case/'volume.inp')
    quality=mesh_quality(coords,conn)
    quality['corner_edge_lengths']=edge_length_statistics(coords,conn[:,:4],
        [(0,1),(1,2),(2,0),(0,3),(1,3),(2,3)])
    source=read(case/'geometry_quality.json')['surface']
    drift=report['volume_mm3']/source['volume_mm3']-1
    mesh_drift=quality['volume_mm3']/report['volume_mm3']-1
    if quality['nonpositive_jacobian_elements']:
        save(receipt,dict(passed=False,failures=['nonpositive Tet10 Jacobian'],surface=report,volume_mesh=quality,
             remesh_volume_drift=drift,volume_mesh_drift=mesh_drift,
             hashes={name:digest(case/name) for name in ['volume.inp','remeshed.stl']}))
        state(case,'MESH_REJECTED');raise RuntimeError('Nonpositive Tet10 Jacobian; see mesh_quality.json')
    grid=pv.UnstructuredGrid(np.column_stack((np.full(len(conn),10),conn)).ravel(),
                            np.full(len(conn),pv.CellType.QUADRATIC_TETRA,dtype=np.uint8),coords)
    boundary=grid.extract_surface(nonlinear_subdivision=1).triangulate().clean()
    boundary.save(case/'fea_surface.stl')
    boundary_report=surface_report(boundary,size)
    failures=[]
    for name,r in [('remeshed',report),('fea_surface',boundary_report)]:
        if r['components']!=1 or r['open_edges']!=0: failures.append(name+' connectivity/manifold')
        if not np.allclose(r['bounds_mm'],[0,size]*3,atol=qcfg['bounds_tolerance_mm'],rtol=0):
            failures.append(name+' dimension drift')
    if abs(drift)>qcfg['volume_drift_fraction']: failures.append('surface remesh volume drift')
    if abs(mesh_drift)>qcfg['volume_drift_fraction']: failures.append('volume mesh volume drift')
    if quality['nonpositive_jacobian_elements']: failures.append('nonpositive Tet10 Jacobian')
    if status['self_intersections'] or status['overlapping_triangles']: failures.append('surface intersections')
    result=dict(passed=not failures,failures=failures,surface=report,fea_surface=boundary_report,
                volume_mesh=quality,remesh_volume_drift=drift,volume_mesh_drift=mesh_drift,
                mass_g=quality['volume_mm3']*cfg['material']['density_tonne_mm3']*1e6,
                hashes={name:digest(case/name) for name in ['volume.inp','remeshed.stl','fea_surface.stl']})
    if (case/'wrapped.stl').exists():
        result['hashes']['wrapped.stl']=digest(case/'wrapped.stl')
    for stage in status.get('surface_stages',[]):
        result['hashes'][stage['file']]=digest(case/stage['file'])
    if failures:
        save(receipt,result)
        state(case,'MESH_REJECTED'); raise RuntimeError('; '.join(failures))
    # Voxel input represents the actual initial FE surface, with explicit cell-centred coordinates.
    n=cfg['geometry']['resolution']; spacing=size/n
    reference=pv.ImageData(dimensions=(n,n,n),spacing=(spacing,)*3,origin=(spacing/2,)*3)
    mask=boundary.voxelize_binary_mask(reference_volume=reference)
    vox=np.asarray(mask.point_data['mask']).reshape((n,n,n),order='F').astype(bool)
    np.savez_compressed(case/'voxel_fea.npz',voxel=vox,origin_mm=np.array([spacing/2]*3),spacing_mm=np.array([spacing]*3))
    areas=vox.sum(axis=(0,1))*spacing**2
    save(case/'geometry_features.json',dict(relative_density=result['volume_mesh']['volume_mm3']/size**3,
         voxel_fraction=float(vox.mean()),surface_area_mm2=float(boundary.area),
         z_area_mm2=areas.tolist(),minimum_voxel_section_mm2=float(areas.min()),
         feature_scope='initial FE geometry only; voxel section is discretized, not exact wall thickness'))
    result['hashes'].update({name:digest(case/name) for name in ['voxel_fea.npz','geometry_features.json']})
    if (case/'midpoint_reconstruction.json').exists():
        result['hashes'].update({name:digest(case/name) for name in ['midpoint_reconstruction.json','volume_3matic.inp']})
    save(receipt,result); state(case,'MESH_OK')
    return result
