"""Build a portable dataset with real numeric sample folders and training files only."""
import argparse
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from src.common import read, save, digest, object_hash, now, safe_id, inside, lock, verify_files
from src.package import stratified_split
from scripts.renumber_training import create_index, export_labels


CONTROL_KEYS = ('weights','density','rotation_deg','topology_names','quaternion_xyzw')
RESULT_FILES = ('curve.npz','stress_strain.csv','metrics.json','quality.json')


def copy_verified(source, target, expected):
    if digest(source) != expected: raise RuntimeError('Source training file changed: '+str(source))
    if target.exists() and digest(target) == expected: return
    temporary = target.with_name(target.name+'.tmp')
    shutil.copy2(source, temporary)
    if digest(temporary) != expected: raise RuntimeError('Copied file differs: '+str(source))
    temporary.replace(target)


def sync_case_result(run, source_id):
    """Publish one completed case without rescanning or copying the other inputs."""
    output = ROOT/'datasets'/run.name
    audit = ROOT/'audits/training_indices'/run.name
    index = read(audit/'training_index.json')
    if index['batch_selection_sha256'] != digest(run/'batch_selection.json'):
        raise RuntimeError('Training numbering differs from the selected batch')
    rows = [r for r in index['samples'] if r['original_sample_id'] == source_id]
    if len(rows) != 1: raise RuntimeError('Case is not uniquely selected: '+source_id)
    row = rows[0]; sid = row['training_id']; case = run/'samples'/source_id
    target = inside(output/'samples'/str(sid), output)
    manifest = read(output/'manifest.json')
    entry = manifest['samples'][sid]
    if entry['sample_id'] != sid or manifest['selection_sha256'] != index['batch_selection_sha256']:
        raise RuntimeError('Clean dataset numbering changed')
    if read(case/'analysis.json')['input_sha256'] != row['input_sha256']:
        raise RuntimeError('FEA input differs from selected training sample')
    extraction = read(case/'extraction.json')
    if not extraction['solver_completed']: raise RuntimeError('Incomplete FEA result: '+source_id)
    exports = read(case/'export_manifest.json'); verify_files(case, exports['hashes'])
    with np.load(case/'curve.npz') as curve:
        if (curve['stress_MPa'].shape != (1001,) or not np.isfinite(curve['stress_MPa']).all()
                or not np.allclose(curve['strain'],np.linspace(0,.05,1001))):
            raise RuntimeError('Invalid complete curve: '+source_id)
    for name in RESULT_FILES:
        copy_verified(case/name, target/name, exports['hashes'][name])
        entry['hashes'][name] = digest(target/name)
    entry.update(labels_ready=True, result_export_sha256=digest(case/'export_manifest.json'),
                 numerical_passed=read(case/'quality.json')['passed'])
    manifest.update(updated_utc=now(), complete_fea_results=sum(r['labels_ready'] for r in manifest['samples']))
    manifest['status'] = 'COMPLETE' if manifest['complete_fea_results']==manifest['sample_count'] else 'INPUTS_READY'
    save(output/'manifest.json', manifest)
    if manifest['status'] == 'COMPLETE': export_labels(run)
    print('Saved FEA result: samples/%d (%d/%d complete)' %
          (sid,manifest['complete_fea_results'],manifest['sample_count']), flush=True)
    return sid


def build_dataset(run, require_results=False):
    output, index = create_index(run); cfg = read(run/'config.json')
    output.mkdir(parents=True, exist_ok=True)
    strata = []
    for row in index['samples']:
        case = Path(row['source_case']); design = read(case/'design.json')
        rho = read(case/'geometry_features.json')['relative_density']
        strata.append(str(design['family'])+':'+str(int(rho/.05)))
    splits = stratified_split(strata, cfg['seed'])
    entries = []; complete = 0
    previous = read(output/'manifest.json') if (output/'manifest.json').exists() else {}
    if previous and previous.get('selection_sha256') != index['batch_selection_sha256']:
        raise RuntimeError('Clean dataset belongs to a different selected batch')
    allowed_root = {'samples','README.md','config.json','manifest.json','labels.npz'}
    unexpected = {p.name for p in output.iterdir()}-allowed_root
    if unexpected: raise RuntimeError('Unexpected files in clean dataset: '+str(sorted(unexpected)))
    for row, split in zip(index['samples'], splits):
        case = Path(row['source_case']); sid = row['training_id']
        target = inside(output/'samples'/str(sid), output); target.mkdir(parents=True, exist_ok=True)
        geometry = read(case/'geometry_quality.json'); mesh = read(case/'mesh_quality.json')
        if not geometry['passed'] or not mesh['passed']: raise RuntimeError('Rejected source: '+str(case))
        verify_files(case, {'inputs.npz':geometry['hashes']['inputs.npz']})
        with np.load(case/'inputs.npz') as source:
            controls = {k:source[k].copy() for k in CONTROL_KEYS}
        if (target/'controls.npz').exists():
            with np.load(target/'controls.npz') as cached:
                if set(cached.files) != set(CONTROL_KEYS) or any(not np.array_equal(cached[k],v) for k,v in controls.items()):
                    raise RuntimeError('Clean control parameters changed: '+str(sid))
        else:
            temporary = target/'controls.npz.tmp'
            with temporary.open('wb') as stream: np.savez_compressed(stream, **controls)
            temporary.replace(target/'controls.npz')
        copy_verified(case/'voxel_fea.npz', target/'voxel.npz', mesh['hashes']['voxel_fea.npz'])
        copy_verified(case/'geometry_features.json', target/'features.json', mesh['hashes']['geometry_features.json'])
        design = read(case/'design.json')
        save(target/'sample.json', dict(sample_id=sid, topology_family=[cfg['geometry']['topologies'][i] for i in design['family']],
            kind=design['kind'], split=str(split), geometry_group=row['geometry_group']))
        labels_ready = False
        if (case/'export_manifest.json').exists() and (case/'extraction.json').exists():
            if read(case/'extraction.json')['solver_completed']:
                verify_files(case, read(case/'export_manifest.json')['hashes'])
                with np.load(case/'curve.npz') as curve:
                    if (curve['stress_MPa'].shape != (1001,) or not np.isfinite(curve['stress_MPa']).all()
                            or not np.allclose(curve['strain'],np.linspace(0,.05,1001))):
                        raise RuntimeError('Invalid complete curve: '+str(case))
                for name in RESULT_FILES: copy_verified(case/name, target/name, digest(case/name))
                labels_ready = True; complete += 1
        if not labels_ready and any((target/name).exists() for name in RESULT_FILES):
            raise RuntimeError('Previously copied result is missing from its source: '+str(sid))
        allowed_sample = {'controls.npz','voxel.npz','features.json','sample.json',*RESULT_FILES}
        unexpected = {p.name for p in target.iterdir()}-allowed_sample
        if unexpected: raise RuntimeError('Unexpected sample files: '+str(sid)+' '+str(sorted(unexpected)))
        entries.append(dict(sample_id=sid, path='samples/'+str(sid), split=str(split), labels_ready=labels_ready,
            hashes={p.name:digest(p) for p in sorted(target.iterdir()) if p.is_file()}))
        if len(entries)%100 == 0: print('Clean dataset: %d/%d' % (len(entries),index['count']), flush=True)
    directories = {p.name for p in (output/'samples').iterdir()}
    if directories != {str(i) for i in range(index['count'])}: raise RuntimeError('Extra or missing sample directories')
    material_keys = ('E_MPa','nu','density_tonne_mm3','plastic','johnson_cook_damage','damage_displacement_mm')
    save(output/'config.json', dict(sample_count=index['count'], seed=cfg['seed'], geometry=cfg['geometry'],
        material={k:cfg['material'][k] for k in material_keys}, mesh=cfg['mesh'], loading=cfg['loading'], solver=cfg['solver']))
    save(output/'manifest.json', dict(schema_version=1, updated_utc=now(), sample_count=index['count'],
        sample_ids=list(range(index['count'])), complete_fea_results=complete,
        status='COMPLETE' if complete==index['count'] else 'INPUTS_READY',
        selection_sha256=index['batch_selection_sha256'], acceptance_basis='USER_SELECTED_DEFAULT',
        split_counts={s:int(np.sum(splits==s)) for s in ('train','validation','test')}, samples=entries))
    (output/'README.md').write_text(
        '# TPMS训练数据集\n\n'
        '500个样本，目录和sample_id为0–499。样本划分为350个训练、75个验证、75个测试。\n\n'
        '每例的controls.npz保存拓扑权重、密度和取向控制；voxel.npz为实际初始FEA几何，'
        '包含体素、原点和间距；features.json保存初始几何特征；sample.json保存编号、拓扑族和划分。\n\n'
        '完整FEA结果可用时加入curve.npz、stress_strain.csv、metrics.json及quality.json。'
        '全部结果可用后，labels.npz按sample_ids=0..499汇总1001点曲线、E、屈服、W1–W5和SEA。'
        '数值检查标志保留在quality.json及labels.npz中。\n\n'
        '应力与E使用MPa，吸能密度使用MJ/m³，SEA使用J/g。屈服未观察到时为NaN及对应有效性标志。'
        '当前结果完成数见manifest.json；输入准备完成不表示FEA已经完成。\n', encoding='utf-8')
    if complete == index['count']: export_labels(run)
    elif require_results: raise RuntimeError('FEA results ready: %d/%d; inputs are retained' % (complete,index['count']))
    print('Actual sample folders: 0..%d; FEA results: %d/%d' % (index['count']-1,complete,index['count']), flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', default='design500_adaptive025_20261001')
    args = parser.parse_args()
    with lock(ROOT/'datasets/.training_index.lock'):
        print(build_dataset(ROOT/'runs'/safe_id(args.run_id)))


if __name__ == '__main__': main()
