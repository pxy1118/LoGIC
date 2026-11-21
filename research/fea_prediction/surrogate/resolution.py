"""Compare validation performance only after checking resolution experiment controls."""
from pathlib import Path
import csv
from .common import digest, read, save
from .data import cohort_signature
from .inference import evaluate, writable_output


def validate_runs(runs, method):
    if method not in ('fusion_pod', 'fusion_direct'):
        raise ValueError('Resolution comparison requires a voxel-based model')
    if len(runs) < 2:
        raise ValueError('At least two completed resolution runs are required')
    plans, manifests, seen = [], [], set()
    for run in map(Path, runs):
        plan = read(run/'plan.json')
        if read(run/'status.json')['status'] not in ('TRAINED', 'SMOKE_PASSED'):
            raise ValueError('Run is incomplete: ' + str(run))
        snap = Path(plan['snapshot'])
        if digest(snap/'manifest.json') != plan['snapshot_sha256']:
            raise ValueError('Snapshot manifest changed')
        manifest = read(snap/'manifest.json')
        resolution = plan['voxel_size']
        if manifest['voxel_size'] != resolution or resolution in seen:
            raise ValueError('Repeated or inconsistent resolution')
        if method not in plan['config']['methods']:
            raise ValueError('Requested model is absent from a run')
        if plans:
            for key in ('config', 'versions', 'code_hashes', 'smoke'):
                if plan[key] != plans[0][key]:
                    raise ValueError('Unmatched training settings: ' + key)
            if cohort_signature(manifest) != cohort_signature(manifests[0]):
                raise ValueError('Different cohorts/labels/splits; use snapshot --match-snapshot-id')
        seen.add(resolution); plans.append(plan); manifests.append(manifest)
    return plans, manifests


def compare(runs, output, method='fusion_pod', device='cpu'):
    runs = list(map(Path, runs))
    plans, manifests = validate_runs(runs, method)
    output = writable_output(output)
    output.mkdir(parents=True)
    rows = []
    for run, plan in sorted(zip(runs, plans), key=lambda pair: pair[1]['voxel_size']):
        n = plan['voxel_size']
        report = evaluate(run, 'validation', method, device, output/f'v{n}')
        tasks = [read(run/f'{method}_seed{seed}'/'status.json') for seed in plan['config']['seeds']]
        memory = [t.get('peak_cuda_memory_allocated_MiB') for t in tasks]
        rows.append(dict(voxel_size=n, run_id=run.name, validation_n=report['n'],
                         curve_rmse_MPa=report['mean_curve_rmse_MPa'],
                         E_MAE_MPa=report['scalar_MAE']['elastic_modulus_MPa'],
                         yield_MAE_MPa=report['scalar_MAE']['yield_strength_MPa'],
                         W5_MAE_MJ_m3=report['scalar_MAE']['W5_MJ_m3'],
                         SEA_MAE_J_g=report['scalar_MAE']['SEA_J_g'],
                         E_valid_pairs=report['scalar_coverage']['elastic_modulus_MPa']['both_valid'],
                         yield_valid_pairs=report['scalar_coverage']['yield_strength_MPa']['both_valid'],
                         min_completed_epochs=min(t['epoch'] for t in tasks), max_completed_epochs=max(t['epoch'] for t in tasks),
                         parameter_count=tasks[0]['parameter_count'],
                         peak_cuda_memory_allocated_MiB=max(memory) if all(m is not None for m in memory) else None))
    result = dict(status='SMOKE_COMPARISON_ONLY' if plans[0]['smoke'] else 'VALIDATION_COMPARISON',
                  method=method, cohort_sha256=cohort_signature(manifests[0]), test_evaluated=False,
                  config=plans[0]['config'], rows=rows,
                  note='Same cohort, code and training settings. Short runs do not establish the best resolution. Memory is PyTorch allocated memory, not total device usage.')
    save(output/'comparison.json', result)
    with (output/'comparison.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    return result
