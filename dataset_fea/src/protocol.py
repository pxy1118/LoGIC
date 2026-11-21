"""Evidence-based numerical comparisons and production protocol freezing."""
from __future__ import annotations
import copy
import numpy as np
from .common import read,save,digest,verify_files,object_hash,inside,now
from .material import audit_material


AXES={'mesh':['mesh'], 'speed':['loading.mean_speed_mm_s'],
      'contact':['loading.platen_contact','loading.contact_penalty_scale'],
      'mass':['solver.mass_scaling_dt_s','solver.mass_scaling_mode','solver.mass_scaling_frequency'], 'resolution':['geometry.resolution']}


def physics(cfg):
    value={k:copy.deepcopy(cfg[k]) for k in ('geometry','mesh','material','loading','solver','quality')}
    for key in ('source','calibration_status'):value['material'].pop(key,None)
    return value


def effective_config(case):
    analysis=read(case/'analysis.json');cfg=copy.deepcopy(analysis['config'])
    cfg['loading']['mean_speed_mm_s']=analysis['actual_mean_speed_mm_s']
    return cfg


def compare(reference,candidate,axis,output,tolerance=.05):
    reference=inside(reference);candidate=inside(candidate);output=inside(output)
    if axis not in AXES or not 0<tolerance<=.05:raise ValueError('Unknown comparison axis or tolerance greater than 5%')
    cases=[reference,candidate];configs=[effective_config(p) for p in cases]
    for p in cases:
        verify_files(p,read(p/'export_manifest.json')['hashes'])
        if not read(p/'quality.json')['passed']:raise ValueError('Both cases must pass numerical checks')
    if read(reference/'design.json')['design_hash']!=read(candidate/'design.json')['design_hash']:
        raise ValueError('Comparison requires identical design controls')
    if axis!='resolution' and digest(reference/'source.stl')!=digest(candidate/'source.stl'):
        raise ValueError('Initial source geometry differs')
    if axis in ('speed','contact','mass') and digest(reference/'volume.inp')!=digest(candidate/'volume.inp'):
        raise ValueError('Comparison requires identical volume mesh')
    trimmed=[physics(c) for c in configs];full_hashes=[object_hash(v) for v in trimmed]
    if full_hashes[0]==full_hashes[1]:raise ValueError('Comparison settings are identical')
    for value in trimmed:
        for key in AXES[axis]:
            bits=key.split('.');container=value
            for bit in bits[:-1]:container=container[bit]
            container.pop(bits[-1],None)
    if object_hash(trimmed[0])!=object_hash(trimmed[1]):raise ValueError('Other physical settings changed with the tested axis')
    metrics=[read(p/'metrics.json') for p in cases];curves=[np.load(p/'curve.npz')['stress_MPa'] for p in cases]
    errors={'curve_normalized_rmse':float(np.sqrt(np.mean((curves[1]-curves[0])**2))/max(np.max(np.abs(curves[0])),1e-12))}
    for key in ('elastic_modulus_MPa','W5_MJ_m3','yield_strength_MPa'):
        a,b=[m[key] for m in metrics]
        errors[key+'_relative_error']=None if a is None or b is None else abs(b-a)/max(abs(a),1e-12)
    failed=[k for k,v in errors.items() if v is not None and v>tolerance]
    if metrics[0]['yield_status']!=metrics[1]['yield_status']:failed.append('yield_observation_changed')
    result=dict(status='PASS' if not failed else 'FAIL',axis=axis,tolerance=tolerance,errors=errors,failures=failed,
         cases=[str(p) for p in cases],export_hashes=[digest(p/'export_manifest.json') for p in cases],
         physics_hashes=full_hashes,scope='Paired numerical sensitivity; material validity is assessed separately')
    save(output,result);return result


def freeze(run,reports,representatives):
    cfg=read(run/'config.json');manifest=read(run/'manifest.json')
    if manifest['purpose']!='dataset':raise ValueError('Freeze a new production run, preserving pilot histories')
    audit=audit_material(cfg)
    if not audit['passed']:raise ValueError('Material audit has not passed: '+'; '.join(audit['failures']))
    if cfg['loading']['mean_speed_mm_s'] is None:raise ValueError('Set the calibrated speed in the frozen production config')
    target=object_hash(physics(cfg));axes=set();evidence=[]
    for path in reports:
        path=inside(path);report=read(path)
        if report['status']!='PASS' or target not in report['physics_hashes']:raise ValueError('Study does not validate this protocol')
        for folder,h in zip(report['cases'],report['export_hashes']):
            if digest(inside(folder)/'export_manifest.json')!=h:raise ValueError('Study evidence changed')
            verify_files(inside(folder),read(inside(folder)/'export_manifest.json')['hashes'])
        axes.add(report['axis']);evidence.append(dict(path=str(path),sha256=digest(path)))
    required={'mesh','speed','contact','resolution'}
    if cfg['solver']['mass_scaling_dt_s'] is not None:required.add('mass')
    if not required<=axes:raise ValueError('Missing studies: '+str(required-axes))
    designs=set();coverage=set();mixed=False;gradient=False;representative_records=[]
    for folder in representatives:
        case=inside(folder);design=read(case/'design.json')
        verify_files(case,read(case/'export_manifest.json')['hashes'])
        if not read(case/'quality.json')['passed'] or object_hash(physics(effective_config(case)))!=target:
            raise ValueError('Representative case does not pass this exact protocol')
        if not (case/'cleanup_receipt.json').exists() or (case/'analysis.odb').exists():
            raise ValueError('Representative export/cleanup is incomplete')
        designs.add(design['design_hash']);coverage.update(design['family'])
        mixed=mixed or len(design['family'])>1;gradient=gradient or design['max_neighbor_rotation_deg']>1
        representative_records.append(dict(case=str(case),export_sha256=digest(case/'export_manifest.json')))
    if len(designs)<6 or coverage!={0,1,2} or not mixed or not gradient:
        raise ValueError('Need six distinct representative cases covering three topologies, mixtures, and rotation gradients')
    receipt=run/'protocol_acceptance.json'
    if receipt.exists():raise FileExistsError('Protocol already frozen')
    save(receipt,dict(status='ACCEPTED',utc=now(),config_sha256=manifest['config_sha256'],physics_sha256=target,
         mean_speed_mm_s=cfg['loading']['mean_speed_mm_s'],material_audit=audit,studies=evidence,
         representatives=representative_records))
    return receipt
