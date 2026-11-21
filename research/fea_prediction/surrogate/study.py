"""Frozen, resumable paper experiment matrix. Never invokes or changes FEA."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import time
import numpy as np

from .common import DEFAULT_DATASET, ROOT, code_hashes, digest, now, read, save
from .data import cohort_signature, revoxelize, snapshot
from .training import run_lock, runtime_versions, train
from .models import BASELINES


def nested_order(rows, seed):
    """Proportional deficit scheduling yields deterministic nested stratified prefixes."""
    rng = np.random.default_rng(seed)
    groups = defaultdict(list)
    for row in sorted(rows, key=lambda r:r['sample_id']):
        groups[('+'.join(row['topology_family']), row['kind'])].append(row['sample_id'])
    groups = {k:list(rng.permutation(v)) for k,v in sorted(groups.items())}
    used = {k:0 for k in groups}
    result = []
    for step in range(len(rows)):
        key = max((k for k in groups if used[k] < len(groups[k])),
                  key=lambda k:((step+1)*len(groups[k])/len(rows)-used[k], k))
        result.append(int(groups[key][used[key]]))
        used[key] += 1
    return result


def build_tasks(manifest, cfg):
    rows = manifest['samples']
    tr = [r for r in rows if r['split']=='train']
    va = [r for r in rows if r['split']=='validation']
    ids = lambda rr:[r['sample_id'] for r in rr]
    order = nested_order(tr, cfg['subset_seed'])
    tasks = [dict(name='main', kind='main', voxel_size=cfg['voxel_size'],
                  methods=cfg['methods'], seeds=cfg['seeds'])]
    for n in cfg['learning_sizes']:
        if n > len(tr):
            raise ValueError('Learning size exceeds training cohort')
        if n == len(tr):
            continue  # exact three-seed reference reused from main, not the five-seed mean
        tasks.append(dict(name=f'learning_{n}', kind='learning', train_size=n, voxel_size=cfg['voxel_size'],
                          methods=cfg['supplementary_methods'], seeds=cfg['supplementary_seeds'],
                          selection=dict(train=order[:n], validation=ids(va), holdout=[])))
    for resolution in cfg['resolutions']:
        tasks.append(dict(name=f'resolution_{resolution}', kind='resolution', voxel_size=resolution,
                          methods=[cfg['primary_method']], seeds=cfg['supplementary_seeds'],
                          epochs=cfg['resolution_epochs'], patience=cfg['resolution_epochs']))
    for name, predicate in [('triple', lambda r:len(r['topology_family'])==3),
                            ('boundary', lambda r:r['kind']=='boundary')]:
        held = [r for r in tr+va if predicate(r)]
        if not held:
            raise ValueError('Missing held-out group: '+name)
        selection = dict(train=ids([r for r in tr if not predicate(r)]),
                         validation=ids([r for r in va if not predicate(r)]), holdout=ids(held))
        tasks.append(dict(name='holdout_'+name, kind='holdout', voxel_size=cfg['voxel_size'],
                          methods=cfg['supplementary_methods'], seeds=cfg['supplementary_seeds'],
                          selection=selection))
    return tasks, order


def prepare(root, config_path=None, dataset=None):
    root = Path(root).resolve()
    cfg_path = root/'config.json'
    if cfg_path.exists():
        cfg = read(cfg_path)
        if config_path and read(config_path) != cfg:
            raise ValueError('Frozen study config differs; use a new study ID')
    else:
        cfg = read(config_path or ROOT/'configs/paper_fea.json')
        root.mkdir(parents=True, exist_ok=True)
        save(cfg_path, cfg)
        save(root/'source.json', dict(dataset=str(Path(dataset or DEFAULT_DATASET).resolve())))
    source = Path(read(root/'source.json')['dataset'])
    if dataset and Path(dataset).resolve() != source:
        raise ValueError('Frozen source differs')
    base = root/'snapshots'/f"v{cfg['voxel_size']}"
    if not (base/'manifest.json').exists():
        snapshot(source, base, 'simulation', cfg['voxel_size'])
    manifest = read(base/'manifest.json')
    if manifest['split_counts'] != cfg['expected_split_counts']:
        raise ValueError('Unexpected paper cohort counts: '+str(manifest['split_counts']))
    snapshots = {}
    for resolution in sorted(set(cfg['resolutions']+[cfg['voxel_size']])):
        path = root/'snapshots'/f'v{resolution}'
        if not (path/'manifest.json').exists():
            revoxelize(base, path, resolution)
        if cohort_signature(read(path/'manifest.json')) != cohort_signature(manifest):
            raise ValueError('Resolution cohorts differ')
        snapshots[str(resolution)] = dict(path=str(path), manifest_sha256=digest(path/'manifest.json'))
    tasks, order = build_tasks(manifest, cfg)
    protocol = dict(schema_version=1, config_sha256=digest(cfg_path), config=cfg,
                    source_manifest_sha256=manifest['source_manifest_sha256'],
                    cohort_sha256=cohort_signature(manifest), snapshots=snapshots,
                    nested_train_order=order, tasks=tasks, code_hashes=code_hashes(),
                    versions=runtime_versions())
    if (root/'protocol.json').exists():
        if read(root/'protocol.json') != protocol:
            raise ValueError('Study protocol/code/environment changed; create a new study ID')
    else:
        save(root/'protocol.json', protocol)
    return protocol


def check_protocol(root):
    root = Path(root)
    p = read(root/'protocol.json')
    if p['config_sha256'] != digest(root/'config.json') or p['code_hashes'] != code_hashes() or p['versions'] != runtime_versions():
        raise ValueError('Frozen study config/code/environment changed; use a new study ID')
    for snap in p['snapshots'].values():
        if digest(Path(snap['path'])/'manifest.json') != snap['manifest_sha256']:
            raise ValueError('Study snapshot changed')
    return p


def checkpoint_inventory(root, protocol):
    inventory = {}
    for spec in protocol['tasks']:
        run = root/'runs'/spec['name']
        if not (run/'status.json').exists() or read(run/'status.json')['status'] != 'TRAINED':
            raise ValueError('Development task incomplete: '+spec['name'])
        inventory[str((run/'plan.json').resolve())] = digest(run/'plan.json')
        validation = read(run/'validation.json')
        for result in validation['methods'].values():
            for value in result['checkpoints']:
                path = Path(value)
                expected = read(path.parent/'status.json')['best_sha256']
                if digest(path) != expected:
                    raise ValueError('Checkpoint changed')
                inventory[str(path.resolve())] = expected
                predictions = path.parent/'validation_predictions.npz'
                state = read(path.parent/'status.json')
                if digest(predictions) != state['validation_predictions_sha256']:
                    raise ValueError('Validation predictions changed')
                inventory[str(predictions.resolve())] = state['validation_predictions_sha256']
    return inventory


def freeze_test(root):
    root = Path(root)
    p = check_protocol(root)
    frozen = dict(protocol_sha256=digest(root/'protocol.json'),
                  checkpoints=checkpoint_inventory(root, p),
                  primary_method=p['config']['primary_method'])
    path = root/'test_freeze.json'
    if path.exists():
        verify_test_freeze(root)
    else:
        save(path, dict(frozen, created_utc=now()))
    return frozen


def verify_test_freeze(root):
    root = Path(root)
    if not (root/'test_freeze.json').exists():
        raise ValueError('Final test is locked until all development tasks are complete and frozen')
    p = check_protocol(root)
    f = read(root/'test_freeze.json')
    if f['protocol_sha256'] != digest(root/'protocol.json') or f['checkpoints'] != checkpoint_inventory(root,p):
        raise ValueError('Final-test freeze no longer matches study')
    return f


def status(root):
    root = Path(root)
    state = read(root/'status.json') if (root/'status.json').exists() else dict(status='NOT_STARTED')
    if (root/'protocol.json').exists():
        p = read(root/'protocol.json')
        tasks = {}
        total_fits, complete_fits = 0, 0
        for spec in p['tasks']:
            path = root/'runs'/spec['name']/'status.json'
            tasks[spec['name']] = read(path) if path.exists() else dict(status='PENDING')
            current = tasks[spec['name']].get('current_task')
            if current:
                progress = path.parent/current/'status.json'
                if progress.exists():
                    tasks[spec['name']]['progress'] = read(progress)
            total_fits += sum(1 if m in BASELINES else len(spec['seeds']) for m in spec['methods'])
            complete_fits += sum(read(s).get('status')=='COMPLETE' for s in path.parent.glob('*/status.json'))
        state = dict(state, tasks=tasks, completed_fits=complete_fits, total_fits=total_fits,
                     test_frozen=(root/'test_freeze.json').exists(),
                     test_accessed=(root/'test_access.json').exists())
    return state


def run_study(root, config_path=None, dataset=None, budget_minutes=None):
    root = Path(root).resolve()
    if budget_minutes is not None and budget_minutes <= 0:
        raise ValueError('Budget must be positive')
    root.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic()+60*budget_minutes if budget_minutes else float('inf')
    with run_lock(root):
        try:
            save(root/'status.json', dict(status='PREPARING', updated_utc=now()))
            p = prepare(root, config_path, dataset)
            for spec in p['tasks']:
                run = root/'runs'/spec['name']
                if (run/'status.json').exists() and read(run/'status.json')['status']=='TRAINED':
                    continue
                if time.monotonic() >= deadline:
                    save(root/'status.json', dict(status='PARTIAL_NOT_FINAL', current_task=spec['name'], updated_utc=now()))
                    return status(root)
                cfg = dict(p['config']['training'], methods=spec['methods'], seeds=spec['seeds'], study_root=str(root))
                cfg.update({k:spec[k] for k in ('selection','epochs','patience') if k in spec})
                snap = Path(p['snapshots'][str(spec['voxel_size'])]['path'])
                save(root/'status.json', dict(status='TRAINING', current_task=spec['name'], updated_utc=now()))
                print('Study task: '+spec['name'], flush=True)
                budget = max(.001, (deadline-time.monotonic())/60) if budget_minutes else None
                result = train(snap, run, None if run.exists() else cfg,
                               resume=run.exists(), budget_minutes=budget)
                if result['status'] != 'TRAINED':
                    save(root/'status.json', dict(status='PARTIAL_NOT_FINAL', current_task=spec['name'], updated_utc=now()))
                    return status(root)
            from .paper_report import complete_study
            save(root/'status.json', dict(status='REPORTING', updated_utc=now()))
            complete_study(root, deadline)
            return status(root)
        except KeyboardInterrupt:
            save(root/'status.json', dict(status='PARTIAL_NOT_FINAL', reason='Interrupted; resume with study run', updated_utc=now()))
            return status(root)
        except Exception as exc:
            save(root/'status.json', dict(status='FAILED', reason=str(exc), updated_utc=now()))
            raise
