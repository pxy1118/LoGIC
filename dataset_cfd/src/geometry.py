"""Extract the undeformed solid only; never reinterpret a deformed ODB."""
from pathlib import Path
import numpy as np
import pyvista as pv
from .common import read, save, digest, state, ROOT


def read_solid_deck(path):
    nodes, elements, mode, pending = {}, [], None, []
    with Path(path).open(encoding='utf8', errors='strict') as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith('**'):
                continue
            if line.startswith('*'):
                if pending:
                    raise ValueError('Truncated C3D10 element')
                key = line.upper().replace(' ', '')
                mode = None
                if key == '*NODE' or key.startswith('*NODE,'):
                    mode = 'node'
                elif key.startswith('*ELEMENT,') and 'TYPE=C3D10' in key:
                    if 'ELSET=SOLID' not in key:
                        raise ValueError('Unexpected non-SOLID Tet10 block')
                    mode = 'solid'
                elif key.startswith(('*PART', '*INSTANCE', '*INCLUDE')):
                    raise ValueError('Explicit assembly/include handling required')
                continue
            if mode not in ('node', 'solid'):
                continue
            values = [x.strip() for x in line.split(',') if x.strip()]
            if mode == 'node':
                node = int(values[0])
                if node in nodes:
                    raise ValueError('Duplicate node label')
                nodes[node] = [float(x) for x in values[1:4]]
            else:
                pending.extend(int(x) for x in values)
                if len(pending) == 11:
                    elements.append(pending); pending = []
                elif len(pending) > 11:
                    raise ValueError('Invalid Tet10 connectivity')
    if pending or not elements:
        raise ValueError('No complete SOLID Tet10 elements')
    labels = np.array(sorted(nodes), dtype=np.int64)
    xyz = np.array([nodes[int(k)] for k in labels])
    elements = np.asarray(elements, dtype=np.int64)
    conn = np.searchsorted(labels, elements[:, 1:])
    if np.any(conn >= len(labels)) or not np.array_equal(labels[conn], elements[:, 1:]):
        raise ValueError('Missing solid node')
    if not np.isfinite(xyz).all():
        raise ValueError('Non-finite coordinates')
    # Abaqus C3D10 midside ordering: 12,23,31,14,24,34.
    pairs = np.array([[0, 1], [1, 2], [2, 0], [0, 3], [1, 3], [2, 3]])
    max_error = 0.0
    for start in range(0, len(conn), 20000):
        block = xyz[conn[start:start + 20000]]
        error = np.linalg.norm(block[:, 4:] - block[:, pairs].mean(axis=2), axis=2)
        max_error = max(max_error, float(error.max()))
    if max_error > 2e-6:
        raise ValueError('Curved Tet10 requires higher-order surface reconstruction')
    return xyz, conn, max_error


def report(surface):
    surface = surface.triangulate().clean()
    connected = surface.connectivity(extraction_mode='all', label_regions=True)
    return {'points': surface.n_points, 'triangles': surface.n_cells,
            'bounds_mm': list(map(float, surface.bounds)),
            'open_edges': int(surface.n_open_edges),
            'boundary_edges': int(surface.extract_feature_edges(boundary_edges=True,
                 non_manifold_edges=False,feature_edges=False,manifold_edges=False).n_cells),
            'nonmanifold_edges': int(surface.extract_feature_edges(boundary_edges=False,
                 non_manifold_edges=True,feature_edges=False,manifold_edges=False).n_cells),
            'volume_mm3': float(abs(surface.volume)),
            'components': int(np.max(connected.cell_data['RegionId']) + 1)}


def prepare_solid(case, sample_id):
    case = Path(case).resolve()
    if not case.is_relative_to(ROOT):
        raise ValueError('Output outside CFD root')
    case.mkdir(parents=True, exist_ok=True)
    import os
    inventory_path = os.environ.get('CFD_V2_INVENTORY')
    inventory = read(Path(inventory_path) if inventory_path else ROOT / 'plan/source_inventory_20261107.json')
    rows = [r for r in inventory['samples'] if r['sample_id'] == sample_id]
    if len(rows) != 1:
        raise ValueError('Sample is not uniquely mapped')
    row = rows[0]
    solid_source = os.environ.get('CFD_V2_SOLID_SOURCE', 'inp_surface')
    if solid_source == 'remeshed_stl':
        source_path = Path(row['files']['remeshed.stl']['path'])
        method = 'prepared remeshed.stl triangle surface'
    else:
        source_path = Path(row['files']['analysis.inp']['path'])
        method = 'SOLID C3D10 initial external planar faces; R3D4 plates excluded'
    source_hash = digest(source_path)
    receipt = case / 'solid_report.json'
    if receipt.exists():
        old = read(receipt)
        if old['source_inp_sha256'] != source_hash or old['solid_sha256'] != digest(case/'solid_initial.stl'):
            raise RuntimeError('Source or reconstructed solid changed')
        return old
    state(case, 'SOURCE_LOCKED', sample_id=sample_id, source_inp_sha256=source_hash,
          solid_source=solid_source)
    if solid_source == 'remeshed_stl':
        surface = pv.read(source_path).triangulate().clean()
        surface = surface.compute_normals(cell_normals=True, point_normals=False,
                                         consistent_normals=True, auto_orient_normals=True)
        solid_elements = 0
        max_error = 0.0
    else:
        xyz, conn, max_error = read_solid_deck(source_path)
        # Straight-sided Tet10s have the same planar boundary as their corners.
        corners = conn[:, :4]
        grid = pv.UnstructuredGrid(np.column_stack([np.full(len(conn), 4), corners]).ravel(),
                                   np.full(len(conn), pv.CellType.TETRA, dtype=np.uint8), xyz)
        surface = grid.extract_surface().triangulate().clean()
        surface = surface.compute_normals(cell_normals=True, point_normals=False,
                                         consistent_normals=True, auto_orient_normals=True)
        solid_elements = len(conn)
    surface.save(case / 'solid_initial.stl')
    r = report(pv.read(case/'solid_initial.stl'))
    if r['open_edges'] or not np.allclose(r['bounds_mm'], [0, 7.5]*3, atol=1e-5):
        raise RuntimeError('Reconstructed initial solid failed boundary checks')
    # Hash both source and reconstructed output, retaining split and identity.
    provenance = {k: row[k] for k in ['sample_id', 'source_sample_id', 'source_case',
                                    'split', 'geometry_group', 'paired_fea', 'topology_family', 'kind']}
    save(case/'source.json', {**provenance, 'source_inp_sha256': source_hash,
                             'solid_source': solid_source,
                             'source_voxel_sha256': digest(row['files']['voxel_fea.npz']['path'])})
    result = {**r, 'source_inp_sha256': source_hash, 'solid_sha256': digest(case/'solid_initial.stl'),
              'solid_elements': solid_elements, 'max_midside_error_mm': max_error,
              'method': method, 'solid_source': solid_source}
    save(receipt, result)
    state(case, 'SOLID_READY', **result)
    return result
