from __future__ import annotations
import hashlib
import json
import shutil
from pathlib import Path
import numpy as np
from .common import ROOT,read,save,digest,verify_files,safe_id
from .material import audit_material
from .storage import verify_case_files


def canonical_group(voxel):
    # Only rotations about loading z preserve this square-platen compression setup.
    return min(hashlib.sha256(np.ascontiguousarray(np.rot90(voxel,k,axes=(0,1))).tobytes()).hexdigest() for k in range(4))


def stratified_split(strata,seed):
    """Keep exact global 70/15/15 counts while spreading each geometry stratum."""
    rng=np.random.default_rng(seed);n=len(strata);fractions=np.array([.70,.15,.15])
    capacity=np.array([round(.7*n),round(.15*n),n-round(.7*n)-round(.15*n)])
    result=np.empty(n,dtype='<U10');names=np.array(['train','validation','test'])
    buckets={s:[i for i,t in enumerate(strata) if t==s] for s in sorted(set(strata))}
    for _,indices in sorted(buckets.items(),key=lambda pair:len(pair[1])):
        rng.shuffle(indices);assigned=np.zeros(3,dtype=int)
        for j,index in enumerate(indices):
            deficit=fractions*(j+1)-assigned
            deficit[capacity<=0]=-np.inf
            best=int(np.argmax(deficit));result[index]=names[best];capacity[best]-=1;assigned[best]+=1
    return result


def package(run,name):
    manifest=read(run/'manifest.json'); cfg=read(run/'config.json')
    if manifest['purpose']!='dataset':raise ValueError('Calibration runs cannot be published as a training dataset')
    protocol=read(run/'protocol_acceptance.json')
    if protocol.get('status')!='ACCEPTED':raise ValueError('Protocol has not passed calibration')
    if not audit_material(cfg)['passed']:raise ValueError('Material audit failed')
    if protocol.get('config_sha256')!=manifest['config_sha256']:raise ValueError('Protocol configuration changed')
    output=ROOT/'datasets'/safe_id(name)
    if output.exists():raise FileExistsError('Release already exists')
    entries=[];stress=[];weights=[];density=[];rotations=[];metrics=[];groups=[];strata=[];files=[];duplicates=[]
    for row in manifest['samples']:
        case=run/'samples'/row['sample_id']
        if not (case/'quality.json').exists() or not read(case/'quality.json')['passed']:continue
        verify_files(case,read(case/'export_manifest.json')['hashes'])
        verify_case_files(case,read(case/'geometry_quality.json')['hashes'])
        verify_case_files(case,read(case/'mesh_quality.json')['hashes'])
        if not read(case/'geometry_quality.json')['passed'] or not read(case/'mesh_quality.json')['passed']:
            raise ValueError('Rejected geometry/mesh cannot be packaged')
        m=read(case/'metrics.json');inputs=np.load(case/'inputs.npz');curve=np.load(case/'curve.npz')
        voxel=np.load(case/'voxel_fea.npz')['voxel'];group=canonical_group(voxel)
        if group in groups:
            duplicates.append(dict(excluded=row['sample_id'],retained=entries[groups.index(group)],group=group));continue
        entries.append(row['sample_id']);stress.append(curve['stress_MPa']);weights.append(inputs['weights'])
        density.append(inputs['density']);rotations.append(inputs['quaternion_xyzw']);metrics.append(m);groups.append(group)
        rho=read(case/'geometry_features.json')['relative_density']
        strata.append(str(row['family'])+':'+str(int(rho/.05)))
        files.append(dict(sample_id=row['sample_id'],case=str(case),
             export_manifest_sha256=digest(case/'export_manifest.json'),mesh_quality_sha256=digest(case/'mesh_quality.json')))
    if len(entries)!=cfg['samples']:raise ValueError('Need %d valid samples, have %d'%(cfg['samples'],len(entries)))
    splits=stratified_split(strata,cfg['seed'])
    staging=output.with_name(output.name+'.incomplete')
    if staging.exists():raise FileExistsError('Inspect incomplete release before retry')
    staging.mkdir(parents=True)
    for entry in files:
        folder=staging/'inputs'/entry['sample_id'];folder.mkdir(parents=True)
        for filename in ('voxel_fea.npz','geometry_features.json'):
            source=Path(entry['case'])/filename;target=folder/filename;shutil.copy2(source,target)
            entry[filename]=dict(path=target.relative_to(staging).as_posix(),sha256=digest(target))
    save(staging/'protocol.json',dict(config=cfg,acceptance=protocol,generator_hashes=manifest.get('generator_hashes',{}),
                                    pipeline_hashes=manifest.get('pipeline_hashes',{})))
    (staging/'DATASET_CARD.md').write_text(
        '# TPMS compression dataset\n\n'
        'Engineering compression of a 7.5 mm cube to 5%, with 1001 curve points. '
        'Stress and E use MPa; W uses MJ/m3. Yield uses a persistent 0.2% offset intersection. '
        'NaN yield values and yield_valid=False mean yield was not observed or not defined.\n\n'
        'Inputs include design controls and initial FE geometry only. '
        'Voxel files and geometry features are copied into inputs/ for portable model training. '
        'Loading work includes recoverable elastic energy. '\
        'This simulation dataset does not establish experimental Ti6Al4V validity beyond the documented material evidence.\n\n'
        'The frozen split is grouped by exact voxel equality and loading-axis quarter turns, '
        'then stratified by topology family and density. Audit near-duplicates before model evaluation. '
        'Rejected attempts and numerical diagnostics remain in the source run.\n',encoding='utf-8')
    def numeric(key):return np.array([np.nan if m[key] is None else m[key] for m in metrics])
    np.savez_compressed(staging/'labels.npz',sample_ids=np.array(entries),strain=np.linspace(0,.05,1001),
        stress_MPa=np.array(stress),E_MPa=numeric('elastic_modulus_MPa'),yield_MPa=numeric('yield_strength_MPa'),
        yield_valid=np.array([m['valid']['yield_strength'] for m in metrics]),
        W_MJ_m3=np.array([[m['W%d_MJ_m3'%i] for i in range(1,6)] for m in metrics]),
        weights=np.array(weights),density=np.array(density),quaternion_xyzw=np.array(rotations),
        group_ids=np.array(groups),split=splits)
    save(staging/'manifest.json',dict(source_run=str(run),source_manifest_sha256=digest(run/'manifest.json'),
         labels_sha256=digest(staging/'labels.npz'),samples=entries,groups=groups,files=files,excluded_duplicates=duplicates,split_counts={
         k:int(np.sum(splits==k)) for k in ['train','validation','test']},
         limitation='Exact voxel equality and z quarter-turn duplicates rejected; near-duplicate review remains required before model fitting'))
    staging.rename(output)
    return output
