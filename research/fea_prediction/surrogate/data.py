from __future__ import annotations

from collections import Counter
from io import BytesIO
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

from .common import digest, now, read, save

INPUT_FILES = ('sample.json', 'controls.npz', 'features.json', 'voxel.npz')
LABEL_FILES = ('curve.npz', 'metrics.json', 'quality.json')
SPLITS = ('train', 'validation', 'test')
TOPOLOGIES = ['Gyroid', 'Diamond', 'Primitive']
DEFAULT_VOXEL_SIZE = 80


def checked_bytes(path, expected):
    value = path.read_bytes()
    if hashlib.sha256(value).hexdigest() != expected:
        raise ValueError('Published hash mismatch (retry with a new snapshot): ' + str(path))
    return value


def decode_json(data):
    return json.loads(data.decode('utf-8'))


def decode_npz(data):
    with np.load(BytesIO(data), allow_pickle=False) as f:
        return {k: f[k].copy() for k in f.files}


def encode_inputs(blobs, config, voxel_size=DEFAULT_VOXEL_SIZE):
    """All features depend on initial geometry only; never open labels here."""
    c = decode_npz(blobs['controls.npz'])
    names = c['topology_names'].tolist()
    if sorted(names) != sorted(TOPOLOGIES):
        raise ValueError('Unsupported topology names')
    w = c['weights'][..., [names.index(k) for k in TOPOLOGIES]]
    rho = c['density']
    q = c['quaternion_xyzw']
    if w.shape != (3, 3, 3, 3) or rho.shape != (3, 3, 3) or q.shape != (3, 3, 3, 4):
        raise ValueError('Expected 3x3x3 controls')
    if not all(np.isfinite(a).all() for a in (w, rho, q)):
        raise ValueError('Nonfinite controls')
    if np.any(w < 0) or not np.allclose(w.sum(-1), 1, atol=1e-6) or np.any((rho <= 0) | (rho >= 1)):
        raise ValueError('Invalid weights/density')
    if not np.allclose(np.linalg.norm(q, axis=-1), 1, atol=1e-5):
        raise ValueError('Invalid unit quaternions')
    # Rotation matrices remove quaternion sign ambiguity and Euler wrapping.
    rotation = Rotation.from_quat(q.reshape(-1, 4)).as_matrix().reshape(3, 3, 3, 9)
    controls = np.concatenate([w, rho[..., None], rotation], axis=-1).reshape(-1)
    f = decode_json(blobs['features.json'])
    area = np.asarray(f['z_area_mm2'], dtype=float)
    size = config['geometry']['size_mm']
    section = area / size**2
    sampled = np.interp(np.linspace(0, 1, 32), np.linspace(0, 1, len(area)), section)
    geometry = np.r_[f['relative_density'], f['voxel_fraction'], f['surface_area_mm2']/size**2,
                     f['minimum_voxel_section_mm2']/size**2, section.mean(), section.std(), sampled]
    source = decode_npz(blobs['voxel.npz'])
    vox = source['voxel']
    if vox.ndim != 3 or len(set(vox.shape)) != 1 or vox.shape[0] % voxel_size:
        raise ValueError('Voxel dimensions must be cubic and divisible by voxel_size')
    if not np.isin(vox, [0, 1]).all():
        raise ValueError('Expected binary initial voxel')
    spacing = np.asarray(source['spacing_mm'])
    if not np.allclose(spacing * np.array(vox.shape), size):
        raise ValueError('Voxel envelope differs from model protocol')
    if not np.allclose(source['origin_mm'], spacing / 2):
        raise ValueError('Expected voxel centers starting at half spacing')
    block = vox.shape[0] // voxel_size
    # Fractional occupancy, not nearest-neighbor thresholding of thin walls.
    pooled = vox.reshape(voxel_size, block, voxel_size, block, voxel_size, block).mean((1, 3, 5))
    if not all(np.isfinite(a).all() for a in (controls, geometry, pooled)):
        raise ValueError('Nonfinite initial-geometry features')
    return controls.astype('float32'), geometry.astype('float32'), pooled[None].astype('float32')


def read_initial_sample(path, config, voxel_size):
    path = Path(path)
    # Intentionally no metrics, quality or curve file dependency.
    return encode_inputs({n: (path/n).read_bytes() for n in INPUT_FILES if n != 'sample.json'}, config, voxel_size)


def validate_curve(data, metrics):
    x, y = data['strain'], data['stress_MPa']
    if x.shape != (1001,) or y.shape != (1001,) or not np.isfinite(y).all():
        raise ValueError('Expected complete finite 1001-point labels')
    if not np.allclose(x, np.linspace(0, .05, 1001), atol=1e-10, rtol=0):
        raise ValueError('Expected exact 0-5% grid; no padding/extrapolation allowed')
    if not metrics.get('valid', {}).get('curve', False):
        raise ValueError('Invalid source curve flag')
    if abs(float(y[0])) > max(.1, .001 * float(np.max(np.abs(y)))):
        raise ValueError('Nonzero initial stress exceeds tolerance')
    y = y.astype('float32').copy()
    y[0] = 0  # bounded numerical roundoff, recorded in snapshot
    return y


def inventory(dataset):
    """Read published manifest once; new concurrent completions wait for next snapshot."""
    dataset = Path(dataset).resolve()
    manifest_bytes = (dataset/'manifest.json').read_bytes()
    manifest = decode_json(manifest_bytes)
    config_bytes = (dataset/'config.json').read_bytes()
    config = decode_json(config_bytes)
    if config['loading']['max_strain'] != .05:
        raise ValueError('This framework targets the 0-5% protocol')
    records, groups, ids = [], {}, set()
    for entry in manifest['samples']:
        sid = int(entry['sample_id'])
        if sid in ids or entry['path'] != 'samples/' + str(sid):
            raise ValueError('Duplicate ID or noncanonical sample path')
        ids.add(sid)
        path = dataset/entry['path']
        meta = decode_json(checked_bytes(path/'sample.json', entry['hashes']['sample.json']))
        split = entry['split']
        if split not in SPLITS or meta['split'] != split or meta['sample_id'] != sid:
            raise ValueError('Sample identity/split mismatch')
        group = meta['geometry_group']
        if group in groups:
            raise ValueError('Duplicate geometry group (including any cross-split leakage): ' + group)
        groups[group] = split
        records.append((entry, meta))
    if len(records) != manifest['sample_count']:
        raise ValueError('Manifest count mismatch')
    return dataset, manifest_bytes, config_bytes, manifest, config, records


def audit(dataset):
    dataset, mb, cb, manifest, config, records = inventory(dataset)
    counts, failures, kinds, families = Counter(), Counter(), Counter(), Counter()
    for entry, meta in records:
        if not entry['labels_ready']:
            continue
        p = dataset/entry['path']
        q = decode_json(checked_bytes(p/'quality.json', entry['hashes']['quality.json']))
        m = decode_json(checked_bytes(p/'metrics.json', entry['hashes']['metrics.json']))
        validate_curve(decode_npz(checked_bytes(p/'curve.npz', entry['hashes']['curve.npz'])), m)
        counts[entry['split']] += 1
        counts['numerical_passed'] += int(q['passed'])
        failures.update(q['failures'])
        kinds[meta['kind']] += 1
        families['+'.join(meta['topology_family'])] += 1
    return dict(created_utc=now(), dataset=str(dataset), total_inputs=len(records),
                complete_curves=sum(counts[s] for s in SPLITS), split_counts={s:counts[s] for s in SPLITS},
                numerical_passed=counts['numerical_passed'], quality_failures=dict(failures),
                labeled_kinds=dict(kinds), labeled_families=dict(families),
                manifest_sha256=hashlib.sha256(mb).hexdigest(), config_sha256=hashlib.sha256(cb).hexdigest(),
                interpretation='Completed simulation labels; numerical flags are not physical validation.')


def snapshot(dataset, output, quality_policy='strict', voxel_size=DEFAULT_VOXEL_SIZE):
    if voxel_size < 4:
        raise ValueError('voxel_size must be at least 4')
    dataset, mb, cb, manifest, config, records = inventory(dataset)
    output = Path(output).resolve()
    if output == dataset or dataset in output.parents:
        raise ValueError('Never write snapshots into the live dataset')
    if output.exists():
        raise FileExistsError('Snapshot already exists; choose a new ID')
    if quality_policy not in ('strict', 'simulation'):
        raise ValueError('Unknown quality policy')
    accepted, excluded = [], []
    for entry, meta in records:
        if not entry['labels_ready']:
            excluded.append(dict(sample_id=entry['sample_id'], reason='not_published'))
            continue
        q = decode_json(checked_bytes(dataset/entry['path']/'quality.json', entry['hashes']['quality.json']))
        if quality_policy == 'strict' and not q['passed']:
            excluded.append(dict(sample_id=entry['sample_id'], reason='numerical_quality', failures=q['failures']))
            continue
        accepted.append((entry, meta, q))
    if not accepted:
        raise ValueError('No eligible labels. strict requires quality.passed=true; use explicit --quality simulation only to model the current simulation protocol.')
    output.mkdir(parents=True)
    (output/'samples').mkdir()
    # READY is written last. An interrupted snapshot is never trainable.
    (output/'source_manifest.json').write_bytes(mb)
    (output/'config.json').write_bytes(cb)
    samples = []
    for index, (entry, meta, q) in enumerate(accepted):
        p = dataset/entry['path']
        blobs = {n: checked_bytes(p/n, entry['hashes'][n]) for n in INPUT_FILES + LABEL_FILES}
        controls, geometry, voxel = encode_inputs(blobs, config, voxel_size)
        metrics = decode_json(blobs['metrics.json'])
        curve = validate_curve(decode_npz(blobs['curve.npz']), metrics)
        target = output/'samples'/str(entry['sample_id'])
        target.mkdir()
        np.savez_compressed(target/'data.npz', controls=controls, geometry=geometry, voxel=voxel, stress=curve)
        for n in ('sample.json', 'metrics.json', 'quality.json'):
            (target/n).write_bytes(blobs[n])
        samples.append(dict(**meta, numerical_passed=bool(q['passed']), failures=q['failures'],
                            source_hashes=entry['hashes'], hashes={n:digest(target/n) for n in
                            ('data.npz', 'sample.json', 'metrics.json', 'quality.json')}))
        if (index+1)%20 == 0:
            print(f'Snapshot: {index+1}/{len(accepted)}', flush=True)
    counts = Counter(s['split'] for s in samples)
    result = dict(schema_version=1, status='READY', created_utc=now(), source=str(dataset),
                  source_manifest_sha256=hashlib.sha256(mb).hexdigest(), config_sha256=hashlib.sha256(cb).hexdigest(),
                  quality_policy=quality_policy, voxel_size=voxel_size, curve_points=1001,
                  input_schema='controls351_geometry38_fractional_voxel_v1',
                  scope='SIMULATION_PROTOCOL_SURROGATE' if quality_policy=='simulation' else 'NUMERICALLY_SCREENED_SURROGATE',
                  zero_origin_policy='Bounded source roundoff set to zero; no smoothing or monotonicity constraint',
                  split_counts={s:counts[s] for s in SPLITS}, excluded=excluded, samples=samples)
    save(output/'manifest.json', result)
    return result


def cohort_signature(manifest):
    """Resolution-independent identity: same inputs, labels, splits and quality policy."""
    cohort = dict(config_sha256=manifest['config_sha256'], quality_policy=manifest['quality_policy'],
                  input_schema=manifest['input_schema'],
                  samples=[{k:r[k] for k in ('sample_id', 'split', 'geometry_group', 'source_hashes')}
                           for r in sorted(manifest['samples'], key=lambda row: row['sample_id'])])
    return hashlib.sha256(json.dumps(cohort, sort_keys=True).encode('utf-8')).hexdigest()


def revoxelize(reference, output, voxel_size=DEFAULT_VOXEL_SIZE):
    """Rebuild only voxels from original high-resolution inputs, retaining a frozen cohort."""
    if voxel_size < 4:
        raise ValueError('voxel_size must be at least 4')
    reference, output = Path(reference).resolve(), Path(output).resolve()
    original = read(reference/'manifest.json')
    if original['status'] != 'READY':
        raise ValueError('Reference snapshot is not READY')
    source = Path(original['source']).resolve()
    if output == source or source in output.parents or output == reference or reference in output.parents:
        raise ValueError('Never write into the live dataset or the reference snapshot')
    if output.exists():
        raise FileExistsError('Snapshot already exists; choose a new ID')
    config_bytes = checked_bytes(reference/'config.json', original['config_sha256'])
    manifest_bytes = checked_bytes(reference/'source_manifest.json', original['source_manifest_sha256'])
    config = decode_json(config_bytes)
    output.mkdir(parents=True)
    (output/'samples').mkdir()
    (output/'config.json').write_bytes(config_bytes)
    (output/'source_manifest.json').write_bytes(manifest_bytes)
    samples = []
    for index, row in enumerate(original['samples']):
        sid = row['sample_id']
        old, raw = reference/'samples'/str(sid), source/'samples'/str(sid)
        frozen = {n:checked_bytes(old/n, h) for n,h in row['hashes'].items()}
        previous = decode_npz(frozen['data.npz'])
        blobs = {n:checked_bytes(raw/n, row['source_hashes'][n])
                 for n in ('controls.npz', 'features.json', 'voxel.npz')}
        c, g, v = encode_inputs(blobs, config, voxel_size)
        if not np.array_equal(c, previous['controls']) or not np.array_equal(g, previous['geometry']):
            raise ValueError('Non-voxel input representation changed; cannot form matched resolution cohorts')
        target = output/'samples'/str(sid)
        target.mkdir()
        np.savez_compressed(target/'data.npz', controls=c, geometry=g, voxel=v, stress=previous['stress'])
        for name in ('sample.json', 'metrics.json', 'quality.json'):
            (target/name).write_bytes(frozen[name])
        samples.append(dict(row, hashes={n:digest(target/n) for n in row['hashes']}))
        if (index+1)%20 == 0:
            print(f'Revoxelize {voxel_size}^3: {index+1}/{len(original["samples"])}', flush=True)
    result = dict(original, created_utc=now(), voxel_size=voxel_size, samples=samples,
                  reference_snapshot=str(reference), reference_snapshot_sha256=digest(reference/'manifest.json'),
                  cohort_sha256=cohort_signature(original))
    save(output/'manifest.json', result)
    return result


def load_split(snapshot_path, split):
    return load_selected(snapshot_path, None, (split,))


def load_selected(snapshot_path, sample_ids, allowed_splits=('train', 'validation')):
    """Explicit development subsets cannot accidentally include original test labels."""
    root = Path(snapshot_path)
    manifest = read(root/'manifest.json')
    if manifest['status'] != 'READY' or digest(root/'config.json') != manifest['config_sha256']:
        raise ValueError('Incomplete or modified snapshot')
    selected = None if sample_ids is None else set(sample_ids)
    if selected is not None and len(selected) != len(sample_ids):
        raise ValueError('Duplicate selected sample ID')
    rows = [r for r in manifest['samples'] if r['split'] in allowed_splits
            and (selected is None or r['sample_id'] in selected)]
    if selected is not None and selected != {r['sample_id'] for r in rows}:
        raise ValueError('Selection includes unavailable or forbidden split IDs')
    if not rows:
        raise ValueError('No labels for selection: ' + str(allowed_splits))
    values = []
    for r in rows:
        p = root/'samples'/str(r['sample_id'])
        for name, expected in r['hashes'].items():
            if digest(p/name) != expected:
                raise ValueError('Modified frozen sample: ' + str(p/name))
        with np.load(p/'data.npz', allow_pickle=False) as a:
            values.append({k:a[k].copy() for k in a.files})
    result = {k:np.stack([v[k] for v in values]) for k in values[0]}
    result['rows'] = rows
    return result
