from __future__ import annotations
import contextlib
import hashlib
import json
import os
from pathlib import Path
import re
import time

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent


def now():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('w', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def safe_id(name):
    if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}', name):
        raise ValueError('IDs must contain only letters, numbers, underscores and hyphens')
    if name.upper() in {'CON','PRN','AUX','NUL',*[f'COM{i}' for i in range(10)],*[f'LPT{i}' for i in range(10)]}:
        raise ValueError('Reserved Windows path name')
    return name


def inside(path, root=ROOT):
    path, root = Path(path).resolve(), Path(root).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError('Path is outside the permitted workspace: ' + str(path))
    return path


@contextlib.contextmanager
def lock(path):
    """OS-backed lock: released automatically after a crash, including on Windows."""
    import msvcrt
    path = inside(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as f:
        if f.tell() == 0:
            f.write(b'0'); f.flush()
        f.seek(0)
        try:
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            raise RuntimeError('Another pipeline process owns ' + str(path)) from exc
        try:
            yield
        finally:
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)


def state(case, stage, **values):
    path = Path(case) / 'state.json'
    data = read(path) if path.exists() else {'created_utc': now(), 'stages': {}}
    data.update(stage=stage, updated_utc=now())
    data['stages'][stage] = dict(utc=now(), **values)
    save(path, data)


def verify_files(folder, hashes):
    for name, expected in hashes.items():
        path = inside(Path(folder) / name,folder)
        if not path.is_file() or digest(path) != expected:
            raise RuntimeError('Missing or changed artifact: ' + str(path))


def load_config(path=None):
    cfg = read(path or ROOT / 'configs/default.json')
    if cfg['geometry']['topologies'] != ['Gyroid', 'Diamond', 'Primitive']:
        raise ValueError('Topology channel order must be G, D, P')
    if cfg['geometry']['rotation_mode'] != 'integrated':
        raise ValueError('Only integrated physical orientation is supported')
    if cfg['geometry'].get('boundary_method','manifold_boolean') not in ('manifold_boolean','implicit_box'):
        raise ValueError('Unknown geometry boundary method')
    resolution=cfg['geometry']['resolution']
    if isinstance(resolution,bool) or not isinstance(resolution,int) or resolution<3:
        raise ValueError('Geometry resolution must be an integer >= 3')
    sigma=cfg['geometry'].get('weight_smoothing_sigma_mm')
    if sigma is not None and (not isinstance(sigma,(int,float)) or not 0<=sigma<float('inf')):
        raise ValueError('Weight smoothing length must be finite and nonnegative')
    cleanup=cfg['geometry'].get('component_cleanup',{})
    if not isinstance(cleanup.get('enabled',False),bool) or not 0<=cleanup.get('max_removed_volume_fraction',.01)<1:
        raise ValueError('Invalid component cleanup policy')
    if cfg['solver']['cpus'] != 16 or cfg['solver']['domains'] != 16:
        raise ValueError('This dataset protocol requires 16 CPUs and domains')
    if cfg['geometry']['size_mm'] != 7.5 or cfg['loading']['max_strain'] != 0.05:
        raise ValueError('This dataset protocol requires a 7.5 mm cube and 5% compression')
    for key in ('surface_edge_mm', 'volume_edge_mm'):
        if not 0 < cfg['mesh'][key] <= 1:
            raise ValueError('Invalid mesh edge length')
    if cfg['mesh'].get('remesh_method','adaptive') not in ('adaptive','expert','uniform','sequence'):
        raise ValueError('Unknown remesh method')
    if cfg['mesh'].get('remesh_method')=='uniform':
        angle=cfg['mesh'].get('surface_sharp_edge_angle_deg',45)
        if angle is not None and not 0<=angle<=180:raise ValueError('Invalid sharp edge angle')
        if not 0<cfg['mesh'].get('surface_split_edge_factor',.4)<=1:raise ValueError('Invalid uniform split factor')
    if cfg['mesh'].get('midside_policy','native') not in ('native','exact_edge_midpoints'):
        raise ValueError('Unknown Tet10 midside policy')
    if cfg['loading'].get('platen_contact','general') not in ('general','kinematic'):
        raise ValueError('Unknown platen contact formulation')
    lo,hi=cfg['geometry']['density_range']
    if not 0<lo<=hi<1:raise ValueError('Invalid density range')
    phase=cfg['geometry'].get('phase_offset_periods',[0,0,0])
    if len(phase)!=3 or not all(isinstance(v,(float,int)) for v in phase):raise ValueError('Invalid phase offset')
    if cfg['solver']['history_intervals']<4000:raise ValueError('At least 4000 history intervals required')
    return cfg
