"""Remove detached fragments while recording the actual geometric change."""
import numpy as np


def triangle_surface(mesh):
    """Exclude zero-dimensional contour artifacts from solid calculations."""
    import pyvista as pv
    surface=mesh.triangulate().clean()
    if surface.n_lines or surface.n_verts:
        # Cleaning collapsed triangles can produce lines and vertices. These
        # carry no solid volume and are also excluded by the STL exporter.
        surface=pv.PolyData(surface.points,surface.faces)
    if not surface.n_faces_strict:raise ValueError('No triangular surface remains')
    return surface


def cleanup_components(mesh,enabled=False,max_removed_volume_fraction=.01):
    surface=triangle_surface(mesh)
    connected=surface.connectivity()
    regions=np.asarray(connected.cell_data['RegionId'],dtype=int)
    triangles=connected.faces.reshape(-1,4)[:,1:]
    vertices=np.asarray(connected.points,dtype=np.float64)[triangles]
    signed=np.einsum('ij,ij->i',vertices[:,0],np.cross(vertices[:,1],vertices[:,2]))/6
    volumes=np.abs(np.bincount(regions,weights=signed))
    total=float(volumes.sum());count=len(volumes)
    if total<=0 or not np.isfinite(volumes).all():raise ValueError('Invalid component volumes')
    record=dict(enabled=enabled,method='largest closed component by volume after cube crop',
        before_components=count,before_volume_mm3=total,
        before_component_volumes_mm3=np.sort(volumes)[::-1].tolist(),
        after_components=count,after_volume_mm3=total,removed_volume_mm3=0.,removed_volume_fraction=0.,
        max_removed_volume_fraction=max_removed_volume_fraction)
    if not enabled or count==1:
        record.update(status='NOT_NEEDED' if enabled else 'DISABLED',passed=True)
        return surface,None,record
    largest=int(np.argmax(volumes));removed_volume=total-float(volumes[largest])
    fraction=removed_volume/total
    if fraction>max_removed_volume_fraction:
        record.update(status='REJECTED_LARGE_COMPONENTS',passed=False,
            proposed_removed_volume_mm3=removed_volume,proposed_removed_volume_fraction=fraction)
        return surface,None,record
    keep=regions==largest
    retained=connected.extract_cells(keep).extract_surface().triangulate().clean()
    removed=connected.extract_cells(~keep).extract_surface().triangulate().clean()
    if retained.n_open_edges or removed.n_open_edges:
        raise ValueError('Component extraction produced open edges')
    record.update(status='CLEANED',passed=True,after_components=1,
        after_volume_mm3=float(abs(retained.volume)),removed_volume_mm3=removed_volume,
        removed_volume_fraction=fraction,removed_components=count-1)
    return retained,removed,record
