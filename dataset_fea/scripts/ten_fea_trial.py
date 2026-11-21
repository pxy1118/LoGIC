"""Reuse ten prepared designs for full 5% diagnostic FEA, one 16-CPU job at a time."""
import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.common import read,save,digest,object_hash,verify_files,lock,safe_id,state,now
from src.storage import verify_case_files
from src.material import audit_material
from src import runtime
from src.metrics import postprocess


FAMILIES=((0,),(1,),(2,),(0,1),(0,2),(1,2),(0,1,2))
RECEIPTS=('geometry_quality.json','mesh_quality.json','component_cleanup.json','midpoint_reconstruction.json',
          'mesher_config.json','mesher_result.json','intermediate_cleanup.json','preparation.json')


def select_cases(source):
    available=[]
    for row in read(source/'manifest.json')['samples']:
        case=source/'samples'/row['sample_id']
        if read(case/'state.json')['stage'] not in ('PREPARED','DATACHECK_OK'):continue
        mesh=read(case/'mesh_quality.json')
        if not mesh['passed']:continue
        features=read(case/'geometry_features.json')
        available.append(dict(row,relative_density=features['relative_density'],elements=mesh['volume_mesh']['elements']))
    chosen=[]
    for family in FAMILIES:
        group=[r for r in available if tuple(r['family'])==family]
        if not group:raise RuntimeError('No prepared example for topology family '+str(family))
        chosen.append(min(group,key=lambda r:(r['elements'],r['sample_id'])))
    for family in FAMILIES[:3]:
        group=[r for r in available if tuple(r['family'])==family and r['sample_id'] not in {c['sample_id'] for c in chosen}]
        if not group:raise RuntimeError('Insufficient density coverage')
        chosen.append(min(group,key=lambda r:(abs(r['relative_density']-.45),r['elements'])))
    # Start with the smallest mesh to obtain a useful cost measurement sooner.
    return sorted(chosen,key=lambda r:(r['elements'],r['sample_id']))


def prepare(source,run):
    if run.exists():raise FileExistsError('Trial already exists; run or inspect it instead of overwriting')
    runtime.assert_external_idle()
    original=read(source/'manifest.json');cfg=read(source/'config.json')
    assert digest(source/'config.json')==original['config_sha256']
    verify_files(ROOT,original['pipeline_hashes']);verify_files(source/'protocol/source',original['generator_hashes'])
    selected=select_cases(source)
    for row in selected:
        case=source/'samples'/row['sample_id'];design=read(case/'design.json');declared=design.pop('design_hash')
        if declared!=row['design_hash'] or object_hash(design)!=declared:raise RuntimeError('Design controls changed')
        for name in ('geometry_quality.json','mesh_quality.json'):verify_case_files(case,read(case/name)['hashes'])
        verify_files(case,{'analysis.inp':read(case/'analysis.json')['input_sha256']})
        if read(case/'analysis.json')['config']!=cfg:raise RuntimeError('Prepared physics differs from frozen run')
    run.mkdir(parents=True)
    shutil.copy2(source/'config.json',run/'config.json')
    for name in original['generator_hashes']:
        dest=run/'protocol/source'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source/'protocol/source'/name,dest)
    revision=run/'protocol/revisions'/object_hash(original['pipeline_hashes'])
    for name in original['pipeline_hashes']:
        dest=revision/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dest)
    for row in selected:
        src=source/'samples'/row['sample_id'];dest=run/'samples'/row['sample_id'];dest.mkdir(parents=True)
        names={'design.json','analysis.inp','analysis.json','inputs.npz','voxel_fea.npz','geometry_features.json','remeshed.stl'}
        journal=read(src/'intermediate_cleanup.json')
        names.update(journal['retained_hashes'])
        names.update(name for name in RECEIPTS if (src/name).exists())
        # Reuse a passed datacheck only when it refers to this identical input.
        if (src/'datacheck.json').exists():
            check=read(src/'datacheck.json')
            if check['passed'] and check['input_sha256']==digest(src/'analysis.inp'):
                names.update(name for name in ('datacheck.json','datacheck_logs.zip') if (src/name).exists())
        for name in sorted(names):shutil.copy2(src/name,dest/name)
        save(dest/'reuse.json',dict(source=str(src),utc=now(),hashes={name:digest(dest/name) for name in names},
            source_pipeline_hashes=original['pipeline_hashes'],training_eligible=False))
        for name in ('geometry_quality.json','mesh_quality.json'):verify_case_files(dest,read(dest/name)['hashes'])
        state(dest,'PREPARED',source=str(src),diagnostic_only=True)
    save(run/'manifest.json',dict(schema_version=1,created_utc=now(),run_id=run.name,purpose='calibration',
         diagnostic_only=True,training_eligible=False,count=10,samples=selected,config_sha256=digest(run/'config.json'),
         generator_hashes=original['generator_hashes'],pipeline_hashes=original['pipeline_hashes'],
         runner_sha256=digest(__file__),source_run=str(source),material_audit=audit_material(cfg),
         solve_policy=dict(cpus=16,endpoint_strain=.05,wall_time_limit_minutes=None,allow_cross_night=True,
             authorization='User requested ten full 5% compression cases, allowing the batch to continue across nights'),
         scope='Diagnostic material and numerical trial; completion does not validate material or release training data'))
    save(run/'trial_status.json',dict(status='PREPARED',utc=now(),current=None,completed=[],failures=[]))
    return selected


def solve_case(case,cfg):
    if (case/'cleanup_receipt.json').exists():return
    stage=read(case/'state.json')['stage']
    if stage in ('BUDGET_SUSPEND_REQUESTED','USER_SUSPEND_REQUESTED'):
        runtime.resume(case,cfg,float('inf'))
    else:
        runtime.assert_external_idle()
        runtime.datacheck(case,cfg,float('inf'))
        runtime.solve(case,cfg,float('inf'))
    runtime.extract(case,cfg,float('inf'))
    postprocess(case,cfg)
    runtime.cleanup(case)


def run_trial(run):
    manifest=read(run/'manifest.json');cfg=read(run/'config.json')
    if not manifest.get('diagnostic_only') or manifest['purpose']!='calibration' or manifest['count']!=10:
        raise RuntimeError('This entry point accepts only the prepared ten-case diagnostic batch')
    if digest(run/'config.json')!=manifest['config_sha256'] or digest(__file__)!=manifest['runner_sha256']:
        raise RuntimeError('Trial configuration or runner changed')
    verify_files(ROOT,manifest['pipeline_hashes']);verify_files(run/'protocol/source',manifest['generator_hashes'])
    runtime.WORKER_SOURCE=run/'protocol/revisions'/object_hash(manifest['pipeline_hashes'])/'workers'
    verify_files(runtime.WORKER_SOURCE, {k.split('/',1)[1]:v for k,v in manifest['pipeline_hashes'].items() if k.startswith('workers/')})
    summary=read(run/'trial_status.json');summary.update(status='RUNNING',started_utc=now())
    with lock(run/'.lock'):
        save(run/'trial_status.json',summary)
        try:
            for row in manifest['samples']:
                case=run/'samples'/row['sample_id'];summary['current']=case.name;summary['utc']=now()
                save(run/'trial_status.json',summary)
                verify_case_files(case,read(case/'reuse.json')['hashes'])
                design=read(case/'design.json');declared=design.pop('design_hash')
                if declared!=row['design_hash'] or object_hash(design)!=declared:raise RuntimeError('Design changed')
                print(case.name+' diagnostic FEA, 16 CPUs, full 5% endpoint',flush=True)
                try:
                    solve_case(case,cfg)
                    if case.name not in summary['completed']:summary['completed'].append(case.name)
                except Exception as exc:
                    save(case/'trial_error.json',dict(utc=now(),error=str(exc)))
                    summary['failures'].append(dict(sample=case.name,error=str(exc)))
                    # An active or suspended solver keeps ownership; never launch the next case over it.
                    if isinstance(exc,TimeoutError) or runtime.active_external():raise
                    if (case/'analysis.odb').exists() and not (case/'analysis.lck').exists():
                        try:
                            runtime.extract(case,cfg)
                            if not read(case/'extraction.json')['solver_completed']:runtime.archive_failure(case);runtime.cleanup(case)
                            else:raise RuntimeError('Complete solve export/validation failed; preserve ODB for review')
                        except Exception as export_error:
                            save(case/'diagnostic_export_error.json',dict(error=str(export_error),utc=now()))
                            # Keep its ODB when it cannot be exported safely; stop accumulation.
                            raise
                    print('FAILED '+case.name+': '+str(exc),flush=True)
                save(run/'trial_status.json',summary)
            summary.update(status='COMPLETED_WITH_FAILURES' if summary['failures'] else 'COMPLETED',current=None)
        except BaseException as exc:
            summary.update(status='INTERRUPTED',error=str(exc));raise
        finally:
            summary['utc']=now();save(run/'trial_status.json',summary)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','run','status'])
    parser.add_argument('--run-id',default='night10_gdp_thesis_20261001')
    parser.add_argument('--source-run',default='design500_adaptive025_20261001')
    args=parser.parse_args();run=ROOT/'runs'/safe_id(args.run_id)
    if args.action=='prepare':
        source=ROOT/'runs'/safe_id(args.source_run)
        with lock(ROOT/'runs/.planning.lock'),lock(source/'.lock'):
            print(json.dumps(prepare(source,run),ensure_ascii=False,indent=2))
    elif args.action=='run':run_trial(run)
    else:
        result=read(run/'trial_status.json')
        if result.get('current'):
            case=run/'samples'/result['current'];sta=case/'analysis.sta'
            result['current_stage']=read(case/'state.json')['stage']
            if sta.exists():result['sta_tail']='\n'.join(sta.read_text(errors='replace').splitlines()[-10:])
        print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
