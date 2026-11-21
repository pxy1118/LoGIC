"""Give selected training samples stable IDs 0..N-1 while preserving FEA source IDs."""
import argparse
import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.common import read, save, digest, object_hash, safe_id, now, lock, verify_files, inside
from scripts.dataset_batch import verify_batch


def create_index(run):
    cfg, selected = verify_batch(run)
    output = ROOT/'datasets'/run.name
    audit = ROOT/'audits/training_indices'/run.name
    audit.mkdir(parents=True, exist_ok=True)
    # Move the former index-only files out of the actual dataset directory.
    for name in ('training_index.json', 'training_index.csv', 'index_manifest.json', 'numbering_validation.json'):
        source = inside(output/name)
        destination = inside(audit/name)
        if source.exists():
            if destination.exists():
                if digest(source) != digest(destination): raise RuntimeError('Different archived index: '+name)
                raise RuntimeError('Duplicate index already archived; inspect '+str(source))
            source.replace(destination)
    rows = []
    for index, source in enumerate(selected['samples']):
        case = run/'samples'/source['sample_id']
        design = read(case/'design.json'); h = design.pop('design_hash')
        if h != source['design_hash'] or object_hash(design) != h:
            raise RuntimeError('Design changed: '+case.name)
        analysis = read(case/'analysis.json')
        if analysis['config'] != cfg or analysis['input_sha256'] != source['input_sha256']:
            raise RuntimeError('Selected input configuration changed: '+case.name)
        for name in ('inputs.npz', 'voxel_fea.npz', 'geometry_features.json'):
            if not (case/name).exists(): raise FileNotFoundError(case/name)
        rows.append(dict(training_id=index, original_sample_id=case.name,
            design_hash=h, geometry_group=source['geometry_group'], input_sha256=source['input_sha256'],
            source_case=case.as_posix(), inputs_npz=(case/'inputs.npz').as_posix(),
            voxel_fea_npz=(case/'voxel_fea.npz').as_posix(),
            geometry_features_json=(case/'geometry_features.json').as_posix(),
            curve_npz=(case/'curve.npz').as_posix(), metrics_json=(case/'metrics.json').as_posix(),
            stress_strain_csv=(case/'stress_strain.csv').as_posix()))
    receipt = dict(schema_version=1, created_utc=now(), count=len(rows),
        source_run=run.as_posix(), config_sha256=selected['config_sha256'],
        batch_selection_sha256=digest(run/'batch_selection.json'),
        numbering='training_id is the fixed row order of the selected batch, starting at zero', samples=rows)
    index_path = audit/'training_index.json'
    if index_path.exists():
        previous = read(index_path)
        for key in ('samples', 'batch_selection_sha256', 'config_sha256', 'count'):
            if previous[key] != receipt[key]: raise RuntimeError('Existing training numbering differs: '+key)
        receipt = previous
    else: save(index_path, receipt)
    csv_path = audit/'training_index.csv'
    temporary = csv_path.with_suffix('.csv.tmp')
    with temporary.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    temporary.replace(csv_path)
    save(audit/'index_manifest.json', dict(count=len(rows),
        hashes={p.name:digest(p) for p in (index_path, csv_path)}, source_run=run.as_posix()))
    return output, receipt


def export_labels(run):
    """Export complete selected curves in training-ID order, retaining numerical flags."""
    import numpy as np
    output, index = create_index(run)
    curves = []; metrics = []; quality = []
    for row in index['samples']:
        case = Path(row['source_case'])
        if not (case/'export_manifest.json').exists() or not read(case/'extraction.json')['solver_completed']:
            raise RuntimeError('Full FEA result not ready: '+row['original_sample_id'])
        verify_files(case, read(case/'export_manifest.json')['hashes'])
        with np.load(case/'curve.npz') as arrays:
            strain = arrays['strain']; stress = arrays['stress_MPa']
            if stress.shape != (1001,) or not np.isfinite(stress).all() or not np.allclose(strain, np.linspace(0,.05,1001)):
                raise RuntimeError('Invalid complete curve: '+case.name)
            curves.append(stress.copy())
        metrics.append(read(case/'metrics.json')); quality.append(read(case/'quality.json'))
    def numeric(key):
        return np.array([np.nan if m[key] is None else m[key] for m in metrics])
    target = output/'labels.npz'; temporary = output/'labels.npz.tmp'
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, sample_ids=np.arange(index['count']),
            strain=np.linspace(0,.05,1001), stress_MPa=np.stack(curves),
            E_MPa=numeric('elastic_modulus_MPa'), yield_MPa=numeric('yield_strength_MPa'),
            yield_valid=np.array([m['valid']['yield_strength'] for m in metrics]),
            W_MJ_m3=np.array([[m['W%d_MJ_m3'%i] for i in range(1,6)] for m in metrics]),
            SEA_J_g=numeric('SEA_J_g'), numerical_passed=np.array([q['passed'] for q in quality]))
    temporary.replace(target)
    audit = ROOT/'audits/training_indices'/run.name
    save(audit/'numerical_quality.json', dict(samples=[dict(training_id=i,
        original_sample_id=index['samples'][i]['original_sample_id'], quality=q) for i,q in enumerate(quality)]))
    save(audit/'labels_manifest.json', dict(utc=now(), count=index['count'],
        training_index_object_sha256=object_hash(index), source_sample_ids=[r['original_sample_id'] for r in index['samples']],
        batch_selection_sha256=index['batch_selection_sha256'], config_sha256=index['config_sha256'],
        hashes={'labels.npz':digest(output/'labels.npz')},
        complete_curves=True, acceptance_basis='USER_SELECTED_DEFAULT'))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('index','export-labels'), nargs='?', default='index')
    parser.add_argument('--run-id', default='design500_adaptive025_20261001')
    args = parser.parse_args(); run = ROOT/'runs'/safe_id(args.run_id)
    # A separate lock permits indexing while the FEA coordinator owns its run lock.
    with lock(ROOT/'datasets/.training_index.lock'):
        from scripts.clean_dataset import build_dataset
        if args.action == 'index':
            print(build_dataset(run))
        else: print(build_dataset(run, require_results=True))


if __name__ == '__main__': main()
