"""Reproducible paper statistics and publication figures from saved predictions."""
from __future__ import annotations

import csv
from pathlib import Path
import time
import numpy as np
import torch
from threadpoolctl import threadpool_limits

from .common import digest, now, read, save
from .data import load_selected, load_split, read_initial_sample
from .inference import chosen_checkpoints, derived, predict_checkpoints
from .models import BASELINES, CurveNet, fit_preprocessing, tabular

SCALARS = ['elastic_modulus_MPa', 'yield_strength_MPa', 'peak_stress_MPa',
           *[f'W{i}_MJ_m3' for i in range(1,6)], 'SEA_J_g']


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def paired_interval(a, b, groups, replicates=10000, seed=20261020):
    """Difference A minus B, resampling independent geometry groups, not curve points."""
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or a.ndim != 1 or len(a) != len(groups):
        raise ValueError('Paired structure arrays must align')
    unique = sorted(set(groups))
    diff = a-b
    per_group = np.array([diff[np.array(groups)==g].mean() for g in unique])
    rng = np.random.default_rng(seed)
    means = per_group[rng.integers(len(per_group), size=(replicates,len(per_group)))].mean(1)
    return dict(mean_difference_MPa=float(per_group.mean()),
                ci95_MPa=np.quantile(means,[.025,.975]).tolist(),
                independent_groups=len(unique), replicates=replicates,
                interpretation='A minus B; exploratory paired interval, no multiplicity correction')


def summarize(truth, pred, rows, geometry, physical):
    truth, pred = np.asarray(truth), np.asarray(pred)
    if truth.shape != pred.shape or truth.shape[1:] != (1001,) or not np.isfinite(pred).all():
        raise ValueError('Expected matching finite complete curves')
    errors = pred-truth
    rms = np.sqrt(np.mean(errors**2,axis=1))
    mae = np.mean(np.abs(errors),axis=1)
    relative = np.linalg.norm(errors,axis=1)/np.maximum(np.linalg.norm(truth,axis=1),1e-8)
    denom = float(np.sum((truth-truth.mean())**2))
    records = []
    for i, row in enumerate(rows):
        reference = derived(truth[i], float(geometry[i,0]), physical)
        estimate = derived(pred[i], float(geometry[i,0]), physical)
        record = dict(sample_id=row['sample_id'], geometry_group=row['geometry_group'],
                      topology_family='+'.join(row['topology_family']), kind=row['kind'],
                      relative_density=float(geometry[i,0]), curve_rmse_MPa=float(rms[i]),
                      curve_mae_MPa=float(mae[i]), relative_l2=float(relative[i]),
                      reference_yield_status=reference['yield_status'],
                      predicted_yield_status=estimate['yield_status'])
        for key in SCALARS:
            record['reference_'+key] = reference[key]
            record['predicted_'+key] = estimate[key]
        records.append(record)
    scalars = {}
    for key in SCALARS:
        actual = [r['reference_'+key] for r in records]
        estimate = [r['predicted_'+key] for r in records]
        valid = [(a,b) for a,b in zip(actual,estimate) if a is not None and b is not None]
        errors_scalar = [abs(a-b) for a,b in valid]
        percentage = [abs(a-b)/abs(a)*100 for a,b in valid if abs(a)>1e-8]
        scalars[key] = dict(MAE=float(np.mean(errors_scalar)) if valid else None,
                            MAPE_percent=float(np.mean(percentage)) if percentage else None,
                            reference_valid=sum(a is not None for a in actual),
                            prediction_valid=sum(b is not None for b in estimate),
                            both_valid=len(valid), valid_fraction=len(valid)/len(rows),
                            mape_valid=len(percentage))
    groups = {}
    for key in ('topology_family','kind'):
        groups[key] = {}
        for value in sorted(set(r[key] for r in records)):
            indices = [i for i,r in enumerate(records) if r[key]==value]
            groups[key][value] = dict(n=len(indices),mean_curve_rmse_MPa=float(rms[indices].mean()),
                                      mean_relative_l2=float(relative[indices].mean()))
    result = dict(n=len(rows),mean_curve_rmse_MPa=float(rms.mean()),
                  mean_curve_mae_MPa=float(mae.mean()),mean_relative_l2=float(relative.mean()),
                  global_r2=float(1-np.sum(errors**2)/denom) if denom>0 else None,
                  global_r2_definition='1 - sum over structures and grid of squared errors / sum((truth - grand mean)^2)',
                  scalar_metrics=scalars, groups=groups,
                  scalar_reference='Both reference and prediction evaluated on the identical 1001-point grid',
                  max_origin_abs_MPa=float(np.abs(pred[:,0]).max()),
                  negative_stress_fraction=float(np.mean(pred<0)))
    return result, records


def evaluation_data(root, spec, split):
    p = read(root/'protocol.json')
    snap = Path(p['snapshots'][str(spec['voxel_size'])]['path'])
    if split == 'test':
        from .study import verify_test_freeze
        verify_test_freeze(root)
        # Access is logged BEFORE opening any test labels, including interrupted evaluations.
        if not (root/'test_access.json').exists():
            save(root/'test_access.json', dict(first_access_utc=now(), protocol_sha256=digest(root/'protocol.json'),
                                              reason='Frozen final evaluation; no subsequent tuning'))
        return load_split(snap, 'test')
    if split == 'holdout':
        return load_selected(snap, spec['selection']['holdout'])
    if 'selection' in spec:
        return load_selected(snap, spec['selection']['validation'])
    return load_split(snap, 'validation')


def evaluate_task(root, spec, split, deadline=float('inf')):
    run = root/'runs'/spec['name']
    cfg = read(root/'config.json')
    out = root/'evaluations'/spec['name']/split
    data = None
    physical = read(run/'plan.json')['physical_config']
    for method in spec['methods']:
        target = out/method
        _, paths = chosen_checkpoints(run, method)
        checkpoint_hashes = {str(p):digest(p) for p in paths}
        if (target/'receipt.json').exists():
            receipt = read(target/'receipt.json')
            if receipt['checkpoints'] != checkpoint_hashes:
                raise ValueError('Saved evaluation checkpoint mismatch')
            for name, expected in receipt['files'].items():
                if digest(target/name) != expected:
                    raise ValueError('Saved evaluation was modified')
            continue
        if time.monotonic() >= deadline:
            return False
        if data is None:
            data = evaluation_data(root,spec,split)
        predictions = []
        for path in paths:
            # Saved validation predictions are exactly the selected checkpoint predictions.
            if split=='validation':
                state=read(path.parent/'status.json')
                if digest(path.parent/'validation_predictions.npz') != state['validation_predictions_sha256']:
                    raise ValueError('Cached validation predictions changed')
                with np.load(path.parent/'validation_predictions.npz') as cached:
                    if cached['sample_ids'].tolist() != [r['sample_id'] for r in data['rows']]:
                        raise ValueError('Validation prediction ordering mismatch')
                    predictions.append(cached['stress'].copy())
            else:
                predictions.append(predict_checkpoints([path],data,cfg['training']['device'])[0])
        stack = np.stack(predictions)
        target.mkdir(parents=True,exist_ok=True)
        summaries = []
        all_records = []
        for i, prediction in enumerate([*stack, stack.mean(0)]):
            label = 'ensemble' if i==len(stack) else paths[i].parent.name
            metrics, records = summarize(data['stress'],prediction,data['rows'],data['geometry'],physical)
            summaries.append(dict(member=label, **metrics))
            all_records.extend(dict(member=label,**r) for r in records)
        scalar_seed_statistics = {}
        for key in SCALARS:
            scalar_seed_statistics[key] = {}
            for metric in ('MAE','MAPE_percent'):
                values = [s['scalar_metrics'][key][metric] for s in summaries[:-1]
                          if s['scalar_metrics'][key][metric] is not None]
                scalar_seed_statistics[key][metric] = dict(
                    mean=float(np.mean(values)) if values else None,
                    sd=float(np.std(values,ddof=1)) if len(values)>1 else 0. if values else None,
                    valid_members=len(values))
        single = [s['mean_curve_rmse_MPa'] for s in summaries[:-1]]
        report = dict(method=method,task=spec['name'],split=split,members=summaries,
                      single_seed_rmse_mean_MPa=float(np.mean(single)),
                      single_seed_rmse_sd_MPa=float(np.std(single,ddof=1)) if len(single)>1 else 0.,
                      scalar_seed_statistics=scalar_seed_statistics,
                      ensemble_members=len(stack), ensemble=summaries[-1])
        save(target/'report.json',report)
        write_csv(target/'samples.csv',all_records)
        np.savez_compressed(target/'predictions.npz', sample_ids=[r['sample_id'] for r in data['rows']],
                            geometry_groups=[r['geometry_group'] for r in data['rows']],
                            truth=data['stress'],members=stack,geometry=data['geometry'])
        save(target/'receipt.json',dict(checkpoints=checkpoint_hashes,files={
            name:digest(target/name) for name in ('report.json','samples.csv','predictions.npz')}))
    return True


def pod_diagnostics(root):
    destination = root/'diagnostics/pod_projection.json'
    if destination.exists():
        return
    p = read(root/'protocol.json')
    snap = Path(p['snapshots'][str(p['config']['voxel_size'])]['path'])
    tr, va = load_split(snap,'train'), load_split(snap,'validation')
    results = []
    with threadpool_limits(limits=p['config']['training']['threads']):
        for rank in p['config']['pod_diagnostic_ranks']:
            prep = fit_preprocessing(tr,rank)
            centered = va['stress']/prep['stress_scale']-prep['curve_mean']
            coefficients = centered@np.linalg.pinv(prep['basis'])
            decoded = (prep['curve_mean']+coefficients@prep['basis'])*prep['stress_scale']
            results.append(dict(rank=rank,train_variance_fraction=prep['pod_train_variance_fraction'],
                                validation_projection_rmse_MPa=float(np.sqrt(np.mean((va['stress']-decoded)**2,axis=1)).mean())))
    save(destination,dict(results=results,interpretation='Projection using true validation curves onto a training-only basis; NOT predictive performance'))


def benchmark(root, deadline=float('inf')):
    cfg = read(root/'config.json')
    p = read(root/'protocol.json')
    snap = Path(p['snapshots'][str(cfg['voxel_size'])]['path'])
    run = root/'runs/main'
    rows = [r for r in read(snap/'manifest.json')['samples'] if r['split']=='validation']
    row = rows[len(rows)//2]
    data = load_selected(snap,[row['sample_id']])
    physical = read(run/'plan.json')['physical_config']
    target = root/'benchmark'
    target.mkdir(exist_ok=True)
    warmup, repeats = cfg['benchmark']['warmup'],cfg['benchmark']['repetitions']
    torch.set_num_threads(cfg['training']['threads'])

    def measure(fn, device):
        for _ in range(warmup):
            fn()
        if device=='cuda':
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        elapsed=[]
        for _ in range(repeats):
            if device=='cuda':
                torch.cuda.synchronize()
            start=time.perf_counter()
            fn()
            if device=='cuda':
                torch.cuda.synchronize()
            elapsed.append((time.perf_counter()-start)*1000)
        return dict(median_ms=float(np.median(elapsed)),p95_ms=float(np.quantile(elapsed,.95)),
                    repetitions=repeats,warmup=warmup,
                    peak_cuda_memory_MiB=float(torch.cuda.max_memory_allocated()/2**20) if device=='cuda' else None)

    with torch.inference_mode(), threadpool_limits(limits=cfg['training']['threads']):
        for method in cfg['methods']:
            _, paths = chosen_checkpoints(run,method)
            for device in cfg['benchmark']['devices']:
                if method in BASELINES and device=='cuda':
                    continue  # NumPy/sklearn baselines have no GPU implementation.
                for count in sorted(set([1,len(paths)])):
                    path=target/f'{method}_{device}_{count}.json'
                    if path.exists():
                        continue
                    if time.monotonic()>=deadline:
                        return False
                    models=[]
                    for checkpoint in paths[:count]:
                        cp=torch.load(checkpoint,map_location='cpu',weights_only=False)
                        prep=cp['preprocessing']
                        if method=='mean':
                            models.append(lambda cp=cp:cp['curve'][None])
                        elif method in BASELINES:
                            x=tabular(data,prep,method)
                            models.append(lambda cp=cp,prep=prep,x=x:
                                          (prep['curve_mean']+cp['estimator'].predict(x)@prep['basis'])*prep['stress_scale'])
                        else:
                            model=CurveNet(prep,method).to(device).eval()
                            model.load_state_dict(cp['state_dict'])
                            args=[None if k=='voxel' and not model.use_voxel else torch.as_tensor(data[k],device=device)
                                  for k in ('controls','geometry','voxel')]
                            models.append(lambda model=model,args=args,scale=prep['stress_scale']:model(*args)*scale)
                    def inference():
                        outputs=[f() for f in models]
                        return (np.stack(outputs).mean(0) if method in BASELINES else torch.stack(outputs).mean(0))
                    timing=measure(inference,device)
                    save(path,dict(method=method,device=device,members=count,sample_id=row['sample_id'],
                                   timing_scope='Resident models and encoded resident inputs, batch=1; includes decoding and ensemble averaging',
                                   hardware=torch.cuda.get_device_name(0) if device=='cuda' else 'CPU; '+str(torch.get_num_threads())+' PyTorch threads',
                                   **timing))
                    del models
        path=target/'input_build.json'
        if not path.exists():
            if time.monotonic()>=deadline:
                return False
            source=Path(read(snap/'manifest.json')['source'])/'samples'/str(row['sample_id'])
            timing=measure(lambda:read_initial_sample(source,physical,cfg['voxel_size']),'cpu')
            save(path,dict(**timing,sample_id=row['sample_id'],
                           timing_scope='Read existing controls/features/240-cubed voxel files and encode C/G/V; warm OS cache. Excludes geometry generation, meshing and initial geometry feature extraction.',
                           fea_speedup=None,fea_speedup_reason='No matched end-to-end FEA timing denominator established'))
    return True


def collect(root):
    output=[]
    for path in sorted((root/'evaluations').glob('*/*/*/report.json')):
        receipt=read(path.parent/'receipt.json')
        for name,expected in receipt['files'].items():
            if digest(path.parent/name)!=expected:
                raise ValueError('Saved evaluation was modified: '+str(path.parent/name))
        report=read(path)
        e=report['ensemble']
        output.append(dict(task=report['task'],split=report['split'],method=report['method'],n=e['n'],
                           members=report['ensemble_members'],
                           single_seed_rmse_mean_MPa=report['single_seed_rmse_mean_MPa'],
                           single_seed_rmse_sd_MPa=report['single_seed_rmse_sd_MPa'],
                           ensemble_rmse_MPa=e['mean_curve_rmse_MPa'],
                           ensemble_mae_MPa=e['mean_curve_mae_MPa'],
                           ensemble_relative_l2=e['mean_relative_l2'],ensemble_global_r2=e['global_r2']))
    return output


def save_figure(fig, base):
    base.parent.mkdir(parents=True,exist_ok=True)
    for suffix in ('pdf','svg','png'):
        fig.savefig(base.with_suffix('.'+suffix),dpi=600 if suffix=='png' else 300,bbox_inches='tight')


def figures(root, summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    out=root/'paper/figures'
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    # Vector, code-generated scientific schematics.
    fig,ax=plt.subplots(figsize=(10,4));ax.set(xlim=(0,10),ylim=(0,4));ax.axis('off')
    boxes=[(.1,2.8,2.8,.8,'Control field C (351)\nMLP: 351 → 64 → 32'),
           (.1,1.6,2.8,.8,'Initial geometry G (38)\nMLP: 38 → 32 → 16'),
           (.1,.4,2.8,.8,'240³ → 80³ occupancy V\n3D CNN: 8 / 16 / 24 → 32'),
           (3.7,1.3,2.2,1.5,'Concatenate (80)\nMLP: 80 → 64 → 8\nPOD coefficients'),
           (6.7,1.3,3,1.5,'Training-only POD basis\n1001-point stress–strain curve\nE, yield, W1–W5, SEA')]
    for x,y,w,h,label in boxes:
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.05',facecolor='#edf4f7',edgecolor='#28516b'))
        ax.text(x+w/2,y+h/2,label,ha='center',va='center')
    for y in (3.2,2,.8):
        ax.annotate('',xy=(3.65,2.05),xytext=(2.95,y),arrowprops={'arrowstyle':'->','color':'#28516b'})
    ax.annotate('',xy=(6.65,2.05),xytext=(5.95,2.05),arrowprops={'arrowstyle':'->'})
    save_figure(fig,out/'architecture');plt.close(fig)
    fig,ax=plt.subplots(figsize=(5,4))
    ax.add_patch(plt.Rectangle((.25,.22),.5,.5,fill=False,lw=2))
    ax.plot([.15,.85],[.17,.17],color='black',lw=5);ax.plot([.15,.85],[.77,.77],color='black',lw=5)
    ax.annotate('',xy=(.5,.79),xytext=(.5,.98),arrowprops={'arrowstyle':'->','lw':2})
    ax.text(.5,.47,'7.5 mm cube\n3 periods per axis\nTi–6Al–4V',ha='center',va='center')
    ax.text(.5,.02,'Fixed lower plate; friction coefficient 0.2',ha='center')
    ax.text(.5,1.03,'Smooth-step compression: 0.375 mm in 1 s\nFinal strain 5%; mean speed 0.375 mm/s',ha='center')
    ax.set(xlim=(0,1),ylim=(-.03,1.15));ax.axis('off')
    save_figure(fig,out/'loading');plt.close(fig)
    for split in ('validation','test'):
        rows=[r for r in summary if r['task']=='main' and r['split']==split]
        if not rows:
            continue
        fig,ax=plt.subplots(figsize=(10,4))
        ax.bar(range(len(rows)),[r['single_seed_rmse_mean_MPa'] for r in rows],
               yerr=[r['single_seed_rmse_sd_MPa'] for r in rows],capsize=3,color='#287b91',label='Single-model mean ± seed SD')
        ax.scatter(range(len(rows)),[r['ensemble_rmse_MPa'] for r in rows],color='#b75332',label='Ensemble',zorder=3)
        ax.set_xticks(range(len(rows)),[r['method'] for r in rows],rotation=35,ha='right')
        ax.set(ylabel='Mean structure RMSE (MPa)',title=split.capitalize());ax.legend()
        save_figure(fig,out/f'model_comparison_{split}');plt.close(fig)
        primary=read(root/'config.json')['primary_method']
        directory=root/'evaluations/main'/split/primary
        with np.load(directory/'predictions.npz') as z:
            truth=z['truth'];pred=z['members'].mean(0);std=z['members'].std(0);ids=z['sample_ids']
        rmse=np.sqrt(np.mean((truth-pred)**2,axis=1));order=np.argsort(rmse)
        indices=[order[0],order[len(order)//2],order[-1]]
        fig,axes=plt.subplots(1,3,figsize=(11,3.2));x=np.linspace(0,5,1001)
        for ax,i,label in zip(axes,indices,['Best','Median','Worst']):
            ax.plot(x,truth[i],color='#222222',label='FEA');ax.plot(x,pred[i],color='#287b91',label='Prediction')
            ax.fill_between(x,pred[i]-std[i],pred[i]+std[i],alpha=.2,color='#287b91',label='Seed SD')
            ax.set(xlabel='Compression strain (%)',ylabel='Stress (MPa)',title=f'{label}: ID {ids[i]}, RMSE {rmse[i]:.2f}')
        axes[0].legend(fontsize=7);save_figure(fig,out/f'representative_curves_{split}');plt.close(fig)
        with (directory/'samples.csv').open(encoding='utf-8-sig',newline='') as stream:
            samples=[r for r in csv.DictReader(stream) if r['member']=='ensemble']
        fig,axes=plt.subplots(1,3,figsize=(10,3.3))
        for ax,key,title in zip(axes,['elastic_modulus_MPa','yield_strength_MPa','W5_MJ_m3'],['Elastic modulus (MPa)','Yield strength (MPa)','W5 (MJ/m³)']):
            pairs=[(float(r['reference_'+key]),float(r['predicted_'+key])) for r in samples
                   if r['reference_'+key] and r['predicted_'+key]]
            if pairs:
                a,b=np.array(pairs).T;low=min(a.min(),b.min());high=max(a.max(),b.max())
                ax.scatter(a,b,s=15,alpha=.75);ax.plot([low,high],[low,high],'--',color='gray')
            ax.set(xlabel='FEA '+title,ylabel='Predicted',title=f'Valid pairs {len(pairs)}/{len(samples)}')
        save_figure(fig,out/f'scalar_parity_{split}');plt.close(fig)
    # Learning curves use the same first three seeds at full size as at smaller sizes.
    cfg=read(root/'config.json')
    fig,ax=plt.subplots(figsize=(6,4))
    for method in cfg['supplementary_methods']:
        xs=[];ys=[];errors=[]
        for n in cfg['learning_sizes']:
            task='main' if n==cfg['expected_split_counts']['train'] else f'learning_{n}'
            path=root/'evaluations'/task/'validation'/method/'report.json'
            if not path.exists(): continue
            rr=read(path);members=rr['members'][:len(cfg['supplementary_seeds'])]
            values=[m['mean_curve_rmse_MPa'] for m in members]
            xs.append(n);ys.append(np.mean(values));errors.append(np.std(values,ddof=1))
        if xs: ax.errorbar(xs,ys,yerr=errors,marker='o',capsize=3,label=method)
    if ax.lines:
        ax.set(xlabel='Training structures',ylabel='Validation RMSE (MPa)');ax.legend()
        save_figure(fig,out/'learning_curves')
    plt.close(fig)
    for kind,split in [('resolution','validation'),('holdout','holdout')]:
        rr=[r for r in summary if r['task'].startswith(kind+'_') and r['split']==split]
        if not rr: continue
        fig,ax=plt.subplots(figsize=(8,4))
        ax.bar(range(len(rr)),[r['single_seed_rmse_mean_MPa'] for r in rr],
               yerr=[r['single_seed_rmse_sd_MPa'] for r in rr],capsize=3)
        ax.set_xticks(range(len(rr)),[r['task']+'\n'+r['method'] for r in rr],rotation=25,ha='right')
        ax.set(ylabel='Mean structure RMSE (MPa)')
        save_figure(fig,out/kind);plt.close(fig)


def manuscript(root, summary, comparisons):
    cfg=read(root/'config.json')
    primary=cfg['primary_method']
    test=[r for r in summary if r['task']=='main' and r['split']=='test']
    text="""# Replacement mechanical methods and results

## 2.2.1 Mechanical dataset construction
The mechanical dataset contains 433 completed heterogeneous scaffold simulations, partitioned into 304 training, 67 validation and 62 test structures using the original geometry-group split. The design space comprises Gyroid, Diamond and Primitive fields. A 3 × 3 × 3 control lattice prescribes topology weights, relative-density controls and orientations. The sampled topology assignments are one-hot at the control points and are interpolated spatially; density controls span 0.25–0.50. Initial geometry is represented on a 240³ voxel grid. Fractional occupancy pooling produces the 80³ input used in the primary model. All completed published curves are retained under the dataset acceptance recorded on 20 October 2026. The frozen release retains the original provenance and quality records.

The mechanical dataset and the CFD dataset are separate experimental cohorts. The existing CFD sampling, topology definitions, simulation settings and results are not replaced by the new mechanical cohort.

## 2.2.2 Mechanical finite-element simulations
The specimen is a 7.5 mm cube containing three periods along each axis. The Ti–6Al–4V material has Young's modulus 112 GPa, Poisson's ratio 0.33 and density 4.5 g cm⁻³. The complete constitutive and solver settings are distributed in the frozen configuration. The models use C3D10 elements with an adaptive target edge length of 0.25 mm. The upper plate applies a smooth-step displacement of 0.375 mm over 1 s, corresponding to 5% final compression strain and a mean displacement rate of 0.375 mm s⁻¹. The lower plate is fixed and the contact friction coefficient is 0.2. The geometry generation for this cohort uses the integrated orientation implementation; this implementation must be distinguished from the pointwise rotation expression in the original methods.

The Abaqus/Explicit protocol uses double precision, variable mass scaling with a target increment of 3 × 10⁻⁵ s and update frequency of 100 increments, and 4000 history-output intervals. The tabulated plastic stress–strain data, Johnson–Cook damage parameters and displacement-based damage evolution are included in the accompanying physical-protocol file; these settings are part of the dataset's simulation definition.

The learning target is the engineering stress–strain response sampled at 1001 points between 0 and 5% strain. No peak-removing smoothing or monotonicity constraint is applied. The effective elastic modulus is fitted between 0.05% and 0.25% strain with the frozen linearity and slope-stability checks. Yield strength is the first persistent 0.2% offset intersection; absent or invalid intersections are reported as missing. W1–W5 are loading-work integrals up to 1–5% strain. These quantities are not identified with irreversible dissipation. Specific energy absorption is the 5% loading work divided by initial mass. Reference and predicted derived quantities are evaluated on the same 1001-point grid.

## 2.2.3 Full-curve prediction and experimental protocol
The primary predictor combines a 351-dimensional control representation, a 38-dimensional initial-geometry descriptor and an 80³ fractional-occupancy voxel field. Orientations are encoded by rotation matrices to remove quaternion-sign ambiguity. The control and geometry multilayer perceptrons have widths 351–64–32 and 38–32–16. The voxel encoder contains three stride-two convolution layers with 8, 16 and 24 channels, group normalization and SiLU activations, followed by adaptive pooling and a 32-dimensional projection. The concatenated 80-dimensional representation is mapped through a 64-unit layer to eight POD coefficients. A training-only mean curve and POD basis reconstruct the full response, with a zero-stress origin. POD reduction follows established reduced-order learning practice [1]; the experimental contribution concerns LoGIC representation and its incremental information for heterogeneous scaffold response.

All feature normalization, stress scaling and POD bases are fitted exclusively to the corresponding training subset. The neural models use AdamW, learning rate 0.001, weight decay 0.001, batch size 12 and dropout 0.1. Training runs for at most 400 epochs with patience 60. The loss combines normalized curve MSE, elastic-window MSE weighted by 0.2 and loading-work error weighted by 0.05. The validation mean of per-structure curve RMSE selects the checkpoint. The five independent seeds are 17, 29, 43, 71 and 101. Single-model mean and sample standard deviation are reported separately from the five-model ensemble.

Baselines include a mean curve, density-only POD kernel ridge regression and control-plus-geometry POD kernel ridge regression. The kernel models use eight POD modes; validation selects alpha from {0.001, 0.01, 0.1, 1, 10} and RBF gamma from {0.25, 1, 4} after feature normalization. Geometric descriptors are a substantive comparator, consistent with prior TPMS property-prediction studies [2]. Neural ablations isolate control, voxel and pairwise input combinations, replace spatial controls with per-channel mean and standard deviation, and replace POD reconstruction with direct 1001-point regression. Parameter counts are reported; branch ablations are not claimed to have identical parameter counts.

Supplementary experiments use seeds 17, 29 and 43. Nested topology-and-kind-stratified training sets contain 76, 152, 228 and 304 structures. Resolution comparisons use 40³, 80³ and 120³ inputs from the same source geometries, batch size 12 and 200 epochs. Unseen-combination evaluation withholds 43 triple-topology samples from the development pool, leaving 270 training and 58 validation structures. Strong-gradient evaluation withholds 32 boundary samples, leaving 278 training and 61 validation structures. The latter is a subset-generalization experiment, not an automatic claim of strict out-of-distribution extrapolation.

After all development experiments, the protocol and checkpoints are frozen before evaluating the 62 test structures. The test set is not used for model selection. The primary metric is mean per-structure curve RMSE; MAE, relative L2 error and global R² are supplementary. Global R² pools residuals and deviations from the grand reference mean across structures and grid points; the grid points are not treated as independent experimental units. Exploratory paired 95% confidence intervals use 10,000 bootstrap resamples of geometry groups. These intervals are not adjusted for multiple comparisons. Across-seed spread is not a calibrated predictive interval.

## 3.2.1 Mechanical prediction results
"""
    if test:
        result=next(r for r in test if r['method']==primary)
        best=min(test,key=lambda r:r['ensemble_rmse_MPa'])
        text+=(
            f"The predefined primary model achieved a test single-model RMSE of "
            f"{result['single_seed_rmse_mean_MPa']:.4f} ± {result['single_seed_rmse_sd_MPa']:.4f} MPa "
            f"(mean ± sample SD over five seeds). Its five-model ensemble obtained "
            f"{result['ensemble_rmse_MPa']:.4f} MPa RMSE, {result['ensemble_mae_MPa']:.4f} MPa MAE, "
            f"{100*result['ensemble_relative_l2']:.3f}% relative L2 error and global R² "
            f"{result['ensemble_global_r2']:.5f}. The lowest observed test ensemble RMSE was "
            f"{best['ensemble_rmse_MPa']:.4f} MPa for {best['method']}; this descriptive ranking "
            "does not change the predefined primary model or select new checkpoints.\n\n")
        report=read(root/'evaluations/main/test'/primary/'report.json')
        for key in SCALARS:
            m=report['ensemble']['scalar_metrics'][key]
            if m['MAE'] is not None:
                text+=f"- {key}: MAE {m['MAE']:.5g}, MAPE {m['MAPE_percent']:.3f}% over {m['mape_valid']} nonzero-reference pairs; {m['both_valid']}/62 valid pairs.\n"
        text+="\n"
        for comparison in comparisons:
            if comparison['split']!='test': continue
            ci=comparison['ci95_MPa']
            conclusion=("The interval includes zero and does not establish a stable difference."
                        if ci[0]<=0<=ci[1] else "The interval excludes zero under this exploratory resampling protocol.")
            text+=(f"For {comparison['a']} minus {comparison['b']}, the paired ensemble RMSE difference "
                   f"was {comparison['mean_difference_MPa']:.4f} MPa (95% CI {ci[0]:.4f} to {ci[1]:.4f}). {conclusion}\n\n")
        text+=("The representative curves were selected algorithmically as the best, median and worst test "
               "structures by primary-model ensemble RMSE. Accurate global curves alone do not establish "
               "local deformation mechanisms, buckling locations or geometric continuity.\n\n")
        for kind,split,title in [('resolution','validation','Resolution sensitivity'),
                                 ('holdout','holdout','Held-group generalization')]:
            records=[r for r in summary if r['task'].startswith(kind+'_') and r['split']==split]
            if records:
                text+=f"### {title}\n\n"
                text+="; ".join(f"{r['task']} / {r['method']}: {r['single_seed_rmse_mean_MPa']:.4f} ± "
                                f"{r['single_seed_rmse_sd_MPa']:.4f} MPa (single-seed mean ± SD, n={r['n']})"
                                for r in records)+".\n\n"
        learning=[r for r in summary if r['task'].startswith('learning_') and r['split']=='validation']
        if learning:
            text+="### Sample efficiency\n\n"
            text+="; ".join(f"{r['task']} / {r['method']}: {r['single_seed_rmse_mean_MPa']:.4f} ± "
                            f"{r['single_seed_rmse_sd_MPa']:.4f} MPa" for r in learning)+". "
            text+="The full-size reference in the learning-curve figure uses the same first three seeds, rather than the five-seed aggregate.\n\n"
        text+=f"""## Mechanical abstract replacement
Using 433 completed finite-element simulations, the full-curve surrogate was evaluated on 62 held-out structures. The predefined control–geometry–voxel POD ensemble achieved a mean curve RMSE of {result['ensemble_rmse_MPa']:.3f} MPa and relative L2 error of {100*result['ensemble_relative_l2']:.2f}% over 0–5% compression strain. Controlled representation, resolution, data-efficiency and held-group experiments quantify the contributions and limitations of the LoGIC input representation.

## Mechanical conclusion replacement
The completed mechanical study quantifies the predictive information in LoGIC controls and the incremental contributions of geometry and voxel inputs using a fixed test cohort. The primary ensemble obtained {result['ensemble_rmse_MPa']:.3f} MPa mean test RMSE. Interpretation of branch benefits follows the paired intervals in the supplementary tables; neither an unconditional voxel advantage nor cross-dataset architectural superiority is assumed.
"""
    else:
        text+="**Pending frozen final evaluation. No final-test result or performance claim is populated.**\n"
    text+="""
## Figure captions
- Architecture: initial controls, initial geometry and fractional voxel occupancy are encoded independently and fused to predict coefficients of a training-only POD basis.
- Loading: 7.5 mm specimen under 0.375 mm smooth-step compression over 1 s; the mean velocity is 0.375 mm s⁻¹.
- Model comparison: single-model mean ± seed SD and separately displayed ensemble error. The mean and kernel baselines are deterministic.
- Representative curves: best, median and worst structures selected by ensemble RMSE; shaded bands show one across-seed SD and are not calibrated uncertainty intervals.
- Scalar parity: reference and predicted quantities on the common 1001-point grid; valid-pair counts are displayed.
- Learning curves: three matched seeds at all sample sizes, including the 304-sample reference reused from the main experiment.
- Resolution: matched cohorts, common batch size and 200-epoch budgets. Timing and memory accompany the accuracy table.
- Held groups: unseen triple-topology combinations and boundary-kind structures, evaluated without opening the original final test cohort during development.

## Revision map
- Replace mechanical content in Sections 2.2.1–2.2.3 and 3.2.1, and the mechanical claims in the abstract and conclusion.
- Split shared Table 2 into explicitly identified mechanical and CFD cohorts. Do not change the CFD counts or settings.
- Update the mechanical loading panel in Fig. 2(a), architecture in Fig. 3 and mechanical predictions in Fig. 10; add supplementary figures and renumber references consistently.
- Replace the mechanical entries of Table 3 with measured model sizes and training settings.
- Add an implementation note in Section 2.1.2 identifying the integrated orientation mapping actually used for this cohort.
- Remove inherited CNN/LSTM scores, 40-point output descriptions and unmeasured 10⁴–10⁵ speedup claims from mechanical text.
- Preserve the original PDF. These are replacement text and figure assets, not an edited or re-typeset journal manuscript.

## References
[1] Lu et al., A comprehensive and fair comparison of two neural operators, CMAME (2022), POD formulation in Section 3.1.4. https://arxiv.org/abs/2111.05512
[2] Rodopoulos et al., Machine learning-based mechanics of TPMS architected materials driven by unit-cell geometric features, npj Metamaterials (2026). https://doi.org/10.1038/s44455-026-00026-9
"""
    (root/'paper/manuscript_replacement.md').write_text(text,encoding='utf-8')


def report_existing(root):
    """Only summarize saved predictions; never opens test labels or triggers evaluation."""
    root=Path(root)
    from .study import check_protocol
    p=check_protocol(root)
    summary=collect(root)
    out=root/'paper';out.mkdir(exist_ok=True)
    main_plan=root/'runs/main/plan.json'
    physical=(read(main_plan)['physical_config'] if main_plan.exists()
              else read(Path(p['snapshots'][str(p['config']['voxel_size'])]['path'])/'config.json'))
    save(out/'physical_protocol.json',physical)
    write_csv(out/'summary.csv',summary);save(out/'summary.json',summary)
    scalar_rows=[];group_rows=[];resource_rows=[]
    for path in sorted((root/'evaluations').glob('*/*/*/report.json')):
        r=read(path)
        for member in r['members']:
            for key,m in member['scalar_metrics'].items():
                scalar_rows.append(dict(task=r['task'],split=r['split'],method=r['method'],
                                        member=member['member'],metric=key,**m))
            for group_type,groups in member['groups'].items():
                for name,m in groups.items():
                    group_rows.append(dict(task=r['task'],split=r['split'],method=r['method'],
                                           member=member['member'],group_type=group_type,group=name,**m))
    for path in sorted((root/'runs').glob('*/*/status.json')):
        state=read(path)
        if state.get('status')!='COMPLETE': continue
        resource_rows.append(dict(task=path.parents[1].name,model=path.parent.name,
                                  **{k:state.get(k) for k in ('epoch','best_epoch','parameter_count','training_seconds','peak_cuda_memory_allocated_MiB')}))
    write_csv(out/'scalar_metrics.csv',scalar_rows);write_csv(out/'group_metrics.csv',group_rows)
    write_csv(out/'training_resources.csv',resource_rows)
    write_csv(out/'timing.csv',[dict(name=f.stem,**read(f)) for f in sorted((root/'benchmark').glob('*.json'))])
    comparisons=[]
    pairs=[('fusion_pod','control_pod'),('fusion_pod','controls_only_pod'),
           ('fusion_pod','voxel_only_pod'),('fusion_pod','pod_krr'),('fusion_pod','fusion_direct'),
           ('controls_only_pod','control_stats_pod'),('controls_only_pod','density_pod_krr')]
    for split in ('validation','test'):
        for a,b in pairs:
            paths=[root/'evaluations/main'/split/m/'predictions.npz' for m in (a,b)]
            if not all(path.exists() for path in paths): continue
            with np.load(paths[0]) as za, np.load(paths[1]) as zb:
                if not np.array_equal(za['sample_ids'],zb['sample_ids']) or not np.array_equal(za['truth'],zb['truth']):
                    raise ValueError('Paired comparison mismatch')
                errors=[np.sqrt(np.mean((z['members'].mean(0)-z['truth'])**2,axis=1)) for z in (za,zb)]
                result=paired_interval(*errors,za['geometry_groups'].tolist(),p['config']['bootstrap_replicates'],p['config']['bootstrap_seed'])
            comparisons.append(dict(split=split,a=a,b=b,**result))
    save(out/'paired_comparisons.json',comparisons)
    write_csv(out/'paired_comparisons.csv',[dict(split=r['split'],a=r['a'],b=r['b'],mean_difference_MPa=r['mean_difference_MPa'],
                                              ci_lower_MPa=r['ci95_MPa'][0],ci_upper_MPa=r['ci95_MPa'][1]) for r in comparisons])
    figures(root,summary);manuscript(root,summary,comparisons)
    state=read(root/'status.json') if (root/'status.json').exists() else {}
    report=dict(status=state.get('status','PARTIAL_NOT_FINAL'),evaluations=len(summary),
                output=str(out),test_accessed=(root/'test_access.json').exists(),
                source='All tables and figures are generated from saved predictions; no manual score transcription')
    save(out/'report.json',report)
    (out/'README.md').write_text(
        '# Paper FEA study\n\nStatus: '+report['status']+
        '\n\nSee `manuscript_replacement.md`, `summary.csv`, `scalar_metrics.csv`, `group_metrics.csv`, '
        '`paired_comparisons.csv`, `training_resources.csv`, `timing.csv` and `figures/`.\n\n'
        'All derived metrics use the same 1001-point reference and prediction grid. '
        'Seed SD and bootstrap intervals describe different sources of variation. '
        'No end-to-end FEA speedup is claimed without a matched timing denominator.\n',encoding='utf-8')
    return report


def complete_study(root, deadline=float('inf')):
    from .study import check_protocol, freeze_test, verify_test_freeze
    root=Path(root)
    p=check_protocol(root)
    for spec in p['tasks']:
        for split in (['validation','holdout'] if spec['kind']=='holdout' else ['validation']):
            if not evaluate_task(root,spec,split,deadline):
                save(root/'status.json',dict(status='PARTIAL_NOT_FINAL',current_task='development_evaluation',updated_utc=now()))
                return
    pod_diagnostics(root)
    if time.monotonic()>=deadline:
        save(root/'status.json',dict(status='PARTIAL_NOT_FINAL',current_task='before_final_test',updated_utc=now()))
        return
    freeze_test(root);verify_test_freeze(root)
    if not evaluate_task(root,p['tasks'][0],'test',deadline):
        save(root/'status.json',dict(status='PARTIAL_NOT_FINAL',current_task='final_test_evaluation',updated_utc=now()))
        return
    if not benchmark(root,deadline):
        save(root/'status.json',dict(status='PARTIAL_NOT_FINAL',current_task='benchmark',updated_utc=now()))
        return
    # COMPLETE is only written after all paper outputs render successfully.
    save(root/'status.json',dict(status='REPORTING',updated_utc=now()))
    report_existing(root)
    save(root/'status.json',dict(status='COMPLETE',updated_utc=now(),paper=str(root/'paper')))
    report=read(root/'paper/report.json');report['status']='COMPLETE';save(root/'paper/report.json',report)
    readme=root/'paper/README.md'
    readme.write_text(readme.read_text(encoding='utf-8').replace('Status: REPORTING','Status: COMPLETE'),encoding='utf-8')
