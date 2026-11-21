from __future__ import annotations

from contextlib import contextmanager
import csv
import os
from pathlib import Path
import shutil
import time
import numpy as np
import sklearn
from sklearn.kernel_ridge import KernelRidge
import torch
from threadpoolctl import threadpool_limits

from .common import ROOT, code_hashes, digest, now, read, save
from .data import load_split, load_selected
from .models import BASELINES, BRANCHES, CurveNet, fit_preprocessing, mean_rmse, objective, predict_neural, tabular

DEFAULTS = dict(methods=['mean', 'pod_krr', 'control_pod', 'fusion_pod', 'fusion_direct'],
                seeds=[17, 29, 43], epochs=400, patience=60, batch_size=12, learning_rate=.001,
                weight_decay=.001, rank=8, early_weight=.2, work_weight=.05, threads=2, device='cuda')


def atomic_checkpoint(path, data):
    path = Path(path)
    temp = path.with_suffix('.tmp')
    torch.save(data, temp)
    temp.replace(path)


@contextmanager
def run_lock(run):
    # OS advisory byte lock: released automatically on process death, including Windows.
    path = run/'.writer.lock'
    with path.open('a+b') as stream:
        stream.seek(0)
        if not stream.read(1):
            stream.write(b'0'); stream.flush()
        stream.seek(0)
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == 'nt':
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def runtime_versions():
    return dict(numpy=np.__version__, torch=str(torch.__version__), sklearn=sklearn.__version__)


def save_history(path, history):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['epoch', 'train_loss', 'validation_rmse_MPa', 'seconds'])
        writer.writeheader(); writer.writerows(history)


def payload(method, prep, plan, **extra):
    return dict(method=method, preprocessing=prep, physical_config=plan['physical_config'],
                voxel_size=plan['voxel_size'], scope=plan['scope'], snapshot_sha256=plan['snapshot_sha256'],
                training_sample_ids=plan['training_sample_ids'], training_input_hashes=plan['training_input_hashes'], **extra)


def fit_baseline(method, train, val, task, plan, cfg):
    started = time.monotonic()
    if method == 'mean':
        prep = fit_preprocessing(train, cfg['rank'])
        pred = np.repeat(train['stress'].mean(0)[None], len(val['stress']), axis=0)
        model = payload(method, prep, plan, curve=train['stress'].mean(0))
        search = []
    else:
        best, model, pred, search = float('inf'), None, None, []
        for rank in sorted(set(min(r, len(train['stress'])-1) for r in cfg.get('krr_ranks', (4, 8, 12)))):
            prep = fit_preprocessing(train, rank)
            xt, xv = tabular(train, prep, method), tabular(val, prep, method)
            normalized = train['stress']/prep['stress_scale'] - prep['curve_mean']
            coefficients = normalized @ np.linalg.pinv(prep['basis'])
            for alpha in (.001, .01, .1, 1., 10.):
                for gamma in (.25, 1., 4.):
                    estimator = KernelRidge(alpha=alpha, gamma=gamma, kernel='rbf').fit(xt, coefficients)
                    estimate = (prep['curve_mean']+estimator.predict(xv)@prep['basis'])*prep['stress_scale']
                    score = mean_rmse(val['stress'], estimate)
                    search.append(dict(rank=rank, alpha=alpha, gamma=gamma, validation_rmse_MPa=score))
                    if score < best:
                        best, pred = score, estimate
                        model = payload(method, prep, plan, estimator=estimator, rank=rank, alpha=alpha, gamma=gamma)
        save(task/'search.json', search)
    atomic_checkpoint(task/'best.pt', model)
    np.savez_compressed(task/'validation_predictions.npz', sample_ids=[r['sample_id'] for r in val['rows']], stress=pred)
    score = mean_rmse(val['stress'], pred)
    save(task/'status.json', dict(status='COMPLETE', method=method, validation_rmse_MPa=score,
                                 training_seconds=time.monotonic()-started,
                                 validation_predictions_sha256=digest(task/'validation_predictions.npz'),
                                 best_sha256=digest(task/'best.pt')))
    return pred


def fit_neural(method, seed, train, val, task, plan, cfg, deadline):
    started = time.monotonic()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    prep = fit_preprocessing(train, cfg['rank'])
    device = torch.device(cfg['device'])
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    model = CurveNet(prep, method).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    epoch, history, best, bad = 0, [], float('inf'), 0
    prior_peak = 0.
    last = task/'last.pt'
    if last.exists():
        checkpoint = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['state_dict']); opt.load_state_dict(checkpoint['optimizer'])
        epoch, history, best, bad = checkpoint['epoch'], checkpoint['history'], checkpoint['best'], checkpoint['bad']
        prior_peak = checkpoint.get('peak_cuda_memory_allocated_MiB', 0.)
        torch.set_rng_state(checkpoint['rng_cpu'].cpu())
        if device.type == 'cuda':
            torch.cuda.set_rng_state_all([s.cpu() for s in checkpoint['rng_cuda']])
    # Keep the dataset on CPU; transfer only one batch so higher resolutions do not
    # reserve VRAM for every training structure. Batch size is unchanged across resolutions.
    tensors = {k:torch.as_tensor(train[k]) for k in ('controls', 'geometry', 'voxel', 'stress')}
    target = tensors['stress']/prep['stress_scale']
    parameter_count = sum(p.numel() for p in model.parameters())
    while epoch < cfg['epochs'] and bad < cfg['patience']:
        if time.monotonic() >= deadline:
            save(task/'status.json', dict(status='PARTIAL_NOT_FINAL', epoch=epoch))
            return None
        tick = time.monotonic()
        model.train(); total = 0.
        permutation = torch.randperm(len(target))
        for start in range(0, len(target), cfg['batch_size']):
            ids = permutation[start:start+cfg['batch_size']]
            opt.zero_grad(set_to_none=True)
            pred = model(*(None if k=='voxel' and not model.use_voxel else tensors[k][ids].to(device)
                           for k in ('controls', 'geometry', 'voxel')))
            loss = objective(pred, target[ids].to(device), cfg['early_weight'], cfg['work_weight'])
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training objective')
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            opt.step(); total += float(loss.detach())*len(ids)
        epoch += 1
        prediction = predict_neural(model, val, device, prep['stress_scale'], cfg['batch_size'])
        score = mean_rmse(val['stress'], prediction)
        if not np.isfinite(score):
            raise ValueError('Nonfinite validation score')
        improved = score < best
        best, bad = (score, 0) if improved else (best, bad+1)
        history.append(dict(epoch=epoch, train_loss=total/len(target), validation_rmse_MPa=score,
                            seconds=time.monotonic()-tick))
        state = payload(method, prep, plan, state_dict=model.state_dict(), seed=seed, epoch=epoch,
                        peak_cuda_memory_allocated_MiB=max(prior_peak, float(torch.cuda.max_memory_allocated(device)/2**20)) if device.type=='cuda' else 0.,
                        parameter_count=parameter_count, validation_rmse_MPa=score)
        # last contains best state as well so an interruption between writes is recoverable.
        if improved:
            best_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            best_epoch = epoch
        elif last.exists():
            previous = torch.load(last, map_location='cpu', weights_only=False)
            best_state, best_epoch = previous['best_state'], previous['best_epoch']
        atomic_checkpoint(last, dict(**state, optimizer=opt.state_dict(), history=history, best=best, bad=bad,
                                     best_state=best_state, best_epoch=best_epoch,
                                     rng_cpu=torch.get_rng_state(),
                                     rng_cuda=torch.cuda.get_rng_state_all() if device.type=='cuda' else []))
        atomic_checkpoint(task/'best.pt', dict(state, state_dict=best_state, epoch=best_epoch, validation_rmse_MPa=best))
        save_history(task/'history.csv', history)
        if epoch==1 or epoch%20==0 or epoch==cfg['epochs']:
            save(task/'status.json', dict(status='RUNNING', method=method, seed=seed, epoch=epoch,
                                         best_epoch=best_epoch, validation_rmse_MPa=score,
                                         best_validation_rmse_MPa=best, updated_utc=now()))
            print(f'{method} seed={seed} epoch={epoch} val_RMSE={score:.3f} MPa best={best:.3f}', flush=True)
    if not last.exists():
        return None
    checkpoint = torch.load(last, map_location='cpu', weights_only=False)
    best_model = dict(checkpoint, state_dict=checkpoint['best_state'], epoch=checkpoint['best_epoch'], validation_rmse_MPa=checkpoint['best'])
    for key in ('optimizer', 'history', 'best_state', 'rng_cpu', 'rng_cuda'):
        best_model.pop(key, None)
    atomic_checkpoint(task/'best.pt', best_model)
    model.load_state_dict(best_model['state_dict'])
    pred = predict_neural(model, val, device, prep['stress_scale'], cfg['batch_size'])
    np.savez_compressed(task/'validation_predictions.npz', sample_ids=[r['sample_id'] for r in val['rows']], stress=pred)
    save(task/'status.json', dict(status='COMPLETE', method=method, seed=seed, epoch=epoch, best_epoch=best_model['epoch'],
                                 parameter_count=parameter_count, validation_rmse_MPa=mean_rmse(val['stress'], pred),
                                 invocation_seconds=time.monotonic()-started,
                                 training_seconds=sum(h['seconds'] for h in history),
                                 peak_cuda_memory_allocated_MiB=checkpoint.get('peak_cuda_memory_allocated_MiB') if device.type=='cuda' else None,
                                 validation_predictions_sha256=digest(task/'validation_predictions.npz'),
                                 best_sha256=digest(task/'best.pt')))
    return pred


def train(snapshot, run, config=None, resume=False, budget_minutes=None, smoke=False):
    if budget_minutes is not None and budget_minutes <= 0:
        raise ValueError('budget_minutes must be positive')
    run = Path(run)
    if resume:
        if config is not None or smoke:
            raise ValueError('Resume uses frozen configuration; omit overrides')
        plan = read(run/'plan.json')
        if plan['code_hashes'] != code_hashes() or plan['versions'] != runtime_versions():
            raise ValueError('Code/environment changed; use a new run ID')
        cfg = plan['config']; snapshot = Path(plan['snapshot'])
    else:
        if run.exists():
            raise FileExistsError('Run exists; use --resume or a new ID')
        cfg = dict(DEFAULTS, **(config or {}))
        if smoke:
            cfg.update(epochs=3, patience=3, seeds=[17])
        if (cfg['epochs'] < 1 or cfg['patience'] < 1 or cfg['batch_size'] < 1 or cfg['threads'] < 1
                or cfg['rank'] < 1 or not cfg['seeds'] or not cfg['methods']):
            raise ValueError('Invalid training settings')
        if any(m not in (*BASELINES, *BRANCHES) for m in cfg['methods']):
            raise ValueError('Unsupported method')
        manifest = read(Path(snapshot)/'manifest.json')
        selection = cfg.get('selection', {})
        if selection:
            sets = [set(selection.get(k, [])) for k in ('train', 'validation', 'holdout')]
            if any(sets[i] & sets[j] for i in range(3) for j in range(i)):
                raise ValueError('Overlapping study subsets')
            eligible = {r['sample_id'] for r in manifest['samples'] if r['split'] != 'test'}
            if not set.union(*sets) <= eligible:
                raise ValueError('Study development selection includes original test IDs')
        train_rows = [r for r in manifest['samples'] if
                      (r['sample_id'] in selection['train'] if selection else r['split']=='train')]
        plan = dict(created_utc=now(), snapshot=str(Path(snapshot).resolve()), snapshot_sha256=digest(Path(snapshot)/'manifest.json'),
                    config=cfg, code_hashes=code_hashes(), versions=runtime_versions(), scope=manifest['scope'],
                    voxel_size=manifest['voxel_size'], physical_config=read(Path(snapshot)/'config.json'),
                    training_sample_ids=[r['sample_id'] for r in train_rows],
                    training_input_hashes=[{n:r['source_hashes'][n] for n in ('controls.npz','features.json','voxel.npz')}
                                           for r in train_rows], smoke=smoke)
        run.mkdir(parents=True)
        save(run/'plan.json', plan)
        for relative in plan['code_hashes']:
            dest=run/'source'/relative; dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT/relative, dest)
    if digest(Path(snapshot)/'manifest.json') != plan['snapshot_sha256']:
        raise ValueError('Snapshot manifest changed')
    if cfg['device'].startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; start a new run with --device cpu')
    torch.set_num_threads(cfg['threads'])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # Test labels are never loaded in this function.
    selection = cfg.get('selection')
    if selection:
        tr = load_selected(snapshot, selection['train'])
        va = load_selected(snapshot, selection['validation'])
    else:
        tr, va = load_split(snapshot, 'train'), load_split(snapshot, 'validation')
    if len(tr['stress']) < 8 or len(va['stress']) < 2:
        raise ValueError('Need >=8 train and >=2 validation structures for this experiment')
    deadline = time.monotonic() + budget_minutes*60 if budget_minutes else float('inf')
    with run_lock(run), threadpool_limits(limits=cfg['threads']):
        save(run/'status.json', dict(status='RUNNING', started_utc=now()))
        summary = {}
        try:
            for method in cfg['methods']:
                predictions, checkpoints = [], []
                seeds = [None] if method in BASELINES else cfg['seeds']
                for seed in seeds:
                    name = method if seed is None else f'{method}_seed{seed}'
                    task=run/name; task.mkdir(exist_ok=True)
                    save(run/'status.json', dict(status='RUNNING', current_task=name, updated_utc=now()))
                    status=read(task/'status.json') if (task/'status.json').exists() else {}
                    if status.get('status')=='COMPLETE':
                        if digest(task/'best.pt') != status['best_sha256']:
                            raise ValueError('Completed checkpoint changed')
                        if status.get('validation_predictions_sha256') and digest(task/'validation_predictions.npz') != status['validation_predictions_sha256']:
                            raise ValueError('Completed validation predictions changed')
                        with np.load(task/'validation_predictions.npz') as f:
                            pred=f['stress'].copy()
                    elif time.monotonic() >= deadline:
                        pred=None
                    elif seed is None:
                        pred=fit_baseline(method, tr, va, task, plan, cfg)
                    else:
                        pred=fit_neural(method, seed, tr, va, task, plan, cfg, deadline)
                    if pred is None:
                        save(run/'status.json', dict(status='PARTIAL_NOT_FINAL', current_task=name, updated_utc=now()))
                        return read(run/'status.json')
                    predictions.append(pred); checkpoints.append(str((task/'best.pt').resolve()))
                stack=np.stack(predictions); estimate=stack.mean(0)
                summary[method]=dict(validation_rmse_MPa=mean_rmse(va['stress'], estimate),
                                     single_model_validation_rmse_MPa=[mean_rmse(va['stress'], p) for p in stack],
                                     checkpoints=checkpoints, ensemble_members=len(stack))
                np.savez_compressed(run/(method+'_validation.npz'), sample_ids=[r['sample_id'] for r in va['rows']],
                                    mean=estimate, std=stack.std(0), truth=va['stress'])
            chosen=min(summary, key=lambda m:summary[m]['validation_rmse_MPa'])
            save(run/'validation.json', dict(methods=summary, selected_method=chosen, selection_metric='mean_per_structure_curve_RMSE_MPa',
                                            scope=plan['scope'], test_evaluated=False, smoke=plan['smoke']))
            save(run/'status.json', dict(status='SMOKE_PASSED' if plan['smoke'] else 'TRAINED', selected_method=chosen, updated_utc=now()))
        except KeyboardInterrupt:
            save(run/'status.json', dict(status='PARTIAL_NOT_FINAL', reason='interrupted; resume last complete epoch', updated_utc=now()))
        except Exception as exc:
            save(run/'status.json', dict(status='FAILED', reason=str(exc), updated_utc=now()))
            raise
    return read(run/'status.json')
