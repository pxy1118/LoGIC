from __future__ import annotations

import csv
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits

from .common import PROJECT, digest, now, read, save
from .data import load_split, read_initial_sample
from .metrics import QUALITY_DEFAULTS, curve_metrics
from .models import CurveNet, predict_neural, tabular


def writable_output(path):
    target = Path(path).resolve()
    live = (PROJECT/'dataset_fea').resolve()
    if target == live or live in target.parents:
        raise ValueError('Prediction tools cannot write into dataset_fea')
    if target.exists():
        raise FileExistsError('Output already exists: ' + str(target))
    return target


def derived(stress, relative_density, config):
    cfg = dict(config, quality=QUALITY_DEFAULTS)
    volume = cfg['geometry']['size_mm']**3
    mass_g = relative_density*volume*cfg['material']['density_tonne_mm3']*1e6
    metrics, _, _ = curve_metrics(np.linspace(0, .05, 1001), stress, mass_g, cfg)
    metrics['evaluation_grid'] = '1001-point predicted curve; raw FEA metrics use denser history'
    return metrics


def chosen_checkpoints(run, method=None):
    run = Path(run)
    if read(run/'status.json')['status'] not in ('TRAINED', 'SMOKE_PASSED'):
        raise ValueError('Run has not completed')
    val = read(run/'validation.json')
    method = method or val['selected_method']
    paths = [Path(p) for p in val['methods'][method]['checkpoints']]
    for path in paths:
        if digest(path) != read(path.parent/'status.json')['best_sha256']:
            raise ValueError('Checkpoint hash changed: ' + str(path))
    return method, paths


def predict_checkpoints(paths, data, device='cpu'):
    torch.set_num_threads(2)
    predictions = []
    scope = set()
    for p in paths:
        # Only load locally produced, hash-verified checkpoints. Pickle is not an interchange format.
        cp = torch.load(p, map_location='cpu', weights_only=False)
        prep = cp['preprocessing']; scope.add(cp['snapshot_sha256'])
        if cp['method']=='mean':
            pred = np.repeat(cp['curve'][None], len(data['controls']), axis=0)
        elif cp['method'] in ('pod_krr', 'density_pod_krr'):
            with threadpool_limits(limits=2):
                pred = (prep['curve_mean']+cp['estimator'].predict(tabular(data, prep, cp['method']))@prep['basis'])*prep['stress_scale']
        else:
            model = CurveNet(prep, cp['method']).to(device)
            model.load_state_dict(cp['state_dict'])
            pred = predict_neural(model, data, device, prep['stress_scale'])
        if not np.isfinite(pred).all():
            raise ValueError('Nonfinite model prediction')
        predictions.append(pred)
    if len(scope) != 1:
        raise ValueError('Cannot combine different snapshot protocols')
    stack = np.stack(predictions)
    return stack.mean(0), stack.std(0), cp


def predict(run, sample_dir, output, method=None, device='cpu', input_config=None):
    output = writable_output(output)
    method, paths = chosen_checkpoints(run, method)
    first = torch.load(paths[0], map_location='cpu', weights_only=False)
    sample_dir = Path(sample_dir).resolve()
    config_path = Path(input_config) if input_config else sample_dir.parent.parent/'config.json'
    cfg = read(config_path)
    for key in ('geometry', 'material', 'mesh', 'loading', 'solver'):
        if cfg[key] != first['physical_config'][key]:
            raise ValueError('Input physical protocol differs at: ' + key)
    c, g, v = read_initial_sample(sample_dir, cfg, first['voxel_size'])
    data = dict(controls=c[None], geometry=g[None], voxel=v[None])
    mean, std, cp = predict_checkpoints(paths, data, device)
    output.mkdir(parents=True)
    with (output/'curve.csv').open('w', newline='', encoding='utf-8') as f:
        w=csv.writer(f); w.writerow(['strain', 'stress_MPa', 'ensemble_std_MPa'])
        w.writerows(zip(np.linspace(0,.05,1001), mean[0], std[0]))
    metrics = derived(mean[0], float(g[0]), cfg)
    save(output/'metrics.json', metrics)
    input_hashes={n:digest(sample_dir/n) for n in ('controls.npz','features.json','voxel.npz')}
    known=cp.get('training_input_hashes')
    save(output/'prediction.json', dict(created_utc=now(), method=method, scope=cp['scope'], source_sample=str(sample_dir),
         voxel_size=cp['voxel_size'],
         input_hashes=input_hashes,
         checkpoints={str(p):digest(p) for p in paths}, ensemble_members=len(paths),
         uncertainty='Across-seed standard deviation only; not calibrated confidence or a numerical-quality assessment',
         training_membership='unknown' if known is None else
             ('exact_training_inputs' if input_hashes in known else 'not_exact_training_inputs')))
    plot_curves(output/'curve.png', [mean[0]], None, [std[0]], [sample_dir.name])
    return dict(output=str(output), method=method, scope=cp['scope'])


def plot_curves(path, predictions, truths, stds, titles):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    count=len(predictions); cols=min(3,count); rows=(count+cols-1)//cols
    fig, axes=plt.subplots(rows,cols,figsize=(5*cols,3.6*rows),squeeze=False)
    x=np.linspace(0,5,1001)
    for i, ax in enumerate(axes.flat):
        if i>=count:
            ax.set_visible(False); continue
        if truths is not None:
            ax.plot(x, truths[i], label='FEA', color='#202937', lw=1.5)
        ax.plot(x,predictions[i],label='Prediction',color='#187c91',lw=1.5)
        if stds is not None and np.any(stds[i]>0):
            ax.fill_between(x,predictions[i]-stds[i],predictions[i]+stds[i],alpha=.2,color='#187c91',label='Seed spread (1 SD)')
        ax.set(xlabel='Engineering compression strain (%)',ylabel='Stress (MPa)',title=str(titles[i]))
        ax.grid(alpha=.18); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path,dpi=150); plt.close(fig)


def evaluate(run, split='validation', method=None, device='cpu', output=None):
    run=Path(run); plan=read(run/'plan.json')
    if plan['smoke'] and split=='test':
        raise ValueError('Smoke runs cannot evaluate the held-out test set')
    if split=='test' and plan['config'].get('study_root'):
        from .study import verify_test_freeze
        verify_test_freeze(Path(plan['config']['study_root']))
    snapshot=Path(plan['snapshot'])
    if digest(snapshot/'manifest.json') != plan['snapshot_sha256']:
        raise ValueError('Modified snapshot manifest')
    method, paths=chosen_checkpoints(run,method)
    output=writable_output(output or run/f'evaluation_{split}_{method}')
    data=load_split(snapshot,split)
    pred,std,cp=predict_checkpoints(paths,data,device)
    truth=data['stress']; error=pred-truth
    rms=np.sqrt(np.mean(error**2,axis=1))
    relative=np.linalg.norm(error,axis=1)/np.maximum(np.linalg.norm(truth,axis=1),1e-8)
    scalar_names=['elastic_modulus_MPa','yield_strength_MPa','W1_MJ_m3','W2_MJ_m3','W3_MJ_m3',
                  'W4_MJ_m3','W5_MJ_m3','SEA_J_g','peak_stress_MPa']
    scalar_errors={k:[] for k in scalar_names}; coverage={k:dict(reference_valid=0,prediction_valid=0,both_valid=0) for k in scalar_names}
    samples=[]
    for i,row in enumerate(data['rows']):
        actual=read(snapshot/'samples'/str(row['sample_id'])/'metrics.json')
        estimate=derived(pred[i],float(data['geometry'][i,0]),cp['physical_config'])
        rec=dict(sample_id=row['sample_id'],curve_rmse_MPa=float(rms[i]),relative_l2=float(relative[i]),
                 numerical_passed=row['numerical_passed'],quality_failures=row['failures'],
                 topology_family=row['topology_family'], predicted=estimate)
        for k in scalar_names:
            a,b=actual[k],estimate[k]
            coverage[k]['reference_valid']+=int(a is not None)
            coverage[k]['prediction_valid']+=int(b is not None)
            if a is not None and b is not None:
                coverage[k]['both_valid']+=1; scalar_errors[k].append(abs(a-b))
        samples.append(rec)
    rng=np.random.default_rng(42)
    means=rms[rng.integers(0,len(rms),size=(2000,len(rms)))].mean(1)
    groups={}
    for family in sorted(set('+'.join(r['topology_family']) for r in data['rows'])):
        ids=[i for i,r in enumerate(data['rows']) if '+'.join(r['topology_family'])==family]
        groups[family]=dict(n=len(ids),curve_rmse_MPa=float(rms[ids].mean()))
    report=dict(created_utc=now(),split=split,method=method,n=len(rms),scope=cp['scope'],
                smoke=plan['smoke'],mean_curve_rmse_MPa=float(rms.mean()),median_curve_rmse_MPa=float(np.median(rms)),
                p90_curve_rmse_MPa=float(np.quantile(rms,.9)),mean_relative_l2=float(relative.mean()),
                bootstrap_mean_rmse_ci95_MPa=np.quantile(means,[.025,.975]).tolist(),
                bootstrap_scope='Resampling independent structures in this frozen subset; not FEA validity or cross-dataset uncertainty',
                scalar_MAE={k:float(np.mean(v)) if v else None for k,v in scalar_errors.items()},scalar_coverage=coverage,
                family_metrics=groups,numerical_passed_count=sum(r['numerical_passed'] for r in data['rows']),
                curve_constraints=dict(max_origin_abs_MPa=float(np.abs(pred[:,0]).max()),negative_stress_fraction=float(np.mean(pred<0))),
                scalar_comparison='Predicted 1001-point metrics versus published raw-history FEA metrics; sampling error included',
                uncertainty='Seed spread is not a calibrated prediction interval',checkpoints={str(p):digest(p) for p in paths})
    output.mkdir(parents=True)
    save(output/'report.json',report);save(output/'samples.json',samples)
    np.savez_compressed(output/'predictions.npz',sample_ids=[r['sample_id'] for r in data['rows']],strain=np.linspace(0,.05,1001),
                        prediction=pred,truth=truth,ensemble_std=std)
    order=np.argsort(rms); indices=np.unique(order[np.linspace(0,len(rms)-1,min(6,len(rms))).astype(int)])
    plot_curves(output/'curves.png',pred[indices],truth[indices],std[indices],
                [f"ID {data['rows'][i]['sample_id']} | RMSE {rms[i]:.2f} MPa" for i in indices])
    if split=='test':
        save(run/'test_access.json', dict(accessed_utc=now(),method=method,output=str(output),
             note='Test has been inspected. Subsequent tuning requires a new independent evaluation.'))
    return report
