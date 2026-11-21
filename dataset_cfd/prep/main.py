"""Reproducible TPMS geometry, remesh, Abaqus and label pipeline."""
from __future__ import annotations
import argparse
from pathlib import Path
import json
import sys
import time
import shutil
from src.common import ROOT,load_config,read,save,lock,safe_id,verify_files,state,digest,now,object_hash
from src.design import create_run,generate,extend_run
from src.mesh import validate_mesh
from src.deck import write_deck
from src import runtime
from src.metrics import postprocess
from src.package import package
from src.material import audit_material
from src.protocol import freeze
from src.storage import compact_case


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['plan','extend','generate','mesh','prepare','run','resume','extract','diagnose','cleanup','compact','status','freeze','package'])
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--config',type=Path)
    parser.add_argument('--pilot',action='store_true')
    parser.add_argument('--samples',nargs='+')
    parser.add_argument('--limit',type=int,default=1)
    parser.add_argument('--speed-mm-s',type=float)
    parser.add_argument('--budget-minutes',type=float,default=30)
    parser.add_argument('--release')
    parser.add_argument('--continue-on-error',action='store_true')
    parser.add_argument('--with-datacheck',action='store_true',
                        help='Run Abaqus datacheck during prepare; prepare skips it by default')
    parser.add_argument('--keep-intermediates',action='store_true',
                        help='Keep intermediate meshes for this invocation; calibration runs keep them by default')
    parser.add_argument('--workers',type=int,default=1,help='Geometry producers for prepare; 1 keeps serial preparation')
    parser.add_argument('--prefetch',type=int,default=2,help='Maximum geometry samples queued ahead of meshing')
    parser.add_argument('--geometry-threads',type=int,default=4,help='Native math threads per geometry producer')
    parser.add_argument('--fresh-mesher',action='store_true',help='Start a fresh 3-matic per sample instead of reusing a session during prepare')
    parser.add_argument('--reports',nargs='+',type=Path)
    parser.add_argument('--representatives',nargs='+',type=Path)
    parser.add_argument('--count',type=int,default=50)
    parser.add_argument('--reason',default='')
    args=parser.parse_args()
    if args.fresh_mesher and args.action!='prepare':parser.error('--fresh-mesher is only valid for prepare')
    if args.with_datacheck and args.action!='prepare':
        parser.error('--with-datacheck is only valid for prepare')
    if args.keep_intermediates and args.action not in ('generate','mesh','prepare','run'):
        parser.error('--keep-intermediates is only valid for generate, mesh, prepare or run')
    if not 1<=args.workers<=4 or args.geometry_threads<1 or not 1<=args.prefetch<=8:
        parser.error('workers must be 1-4, geometry-threads positive, and prefetch 1-8')
    if args.workers>1 and (args.action!='prepare' or args.prefetch<args.workers):
        parser.error('Parallel workers require prepare and prefetch >= workers')
    run=ROOT/'runs'/safe_id(args.run_id)
    if args.action=='plan':
        with lock(ROOT/'runs/.planning.lock'):
            folder=create_run(args.run_id,load_config(args.config),args.pilot)
        print(json.dumps({'run':str(folder),'purpose':'calibration' if args.pilot else 'dataset'}));return
    cfg=read(run/'config.json');manifest=read(run/'manifest.json')
    if digest(run/'config.json')!=manifest['config_sha256']:raise RuntimeError('Frozen run configuration changed')
    verify_files(run/'protocol/source',manifest['generator_hashes'])
    pipeline_paths=[ROOT/'main.py',*sorted((ROOT/'src').glob('*.py')),*sorted((ROOT/'workers').glob('*.py'))]
    code_hashes={p.relative_to(ROOT).as_posix():digest(p) for p in pipeline_paths}
    if manifest['purpose']=='dataset' and args.action not in ('status','cleanup'):
        verify_files(ROOT,manifest['pipeline_hashes'])
    revision=run/'protocol/revisions'/object_hash(code_hashes)
    if args.action!='status':
        for name in code_hashes:
            path=revision/name
            if not path.exists():
                path.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,path)
        verify_files(revision,code_hashes)
        runtime.WORKER_SOURCE=revision/'workers'
    rows=manifest['samples']
    if args.samples:
        missing=set(args.samples)-{r['sample_id'] for r in rows}
        if missing:raise ValueError('Unknown sample IDs: '+str(missing))
        rows=[r for r in rows if r['sample_id'] in args.samples]
    if args.action=='status':
        for row in rows:
            case=run/'samples'/row['sample_id'];record=read(case/'state.json')
            progress=''
            if (case/'analysis.sta').exists():
                progress='\n'.join((case/'analysis.sta').read_text(errors='replace').splitlines()[-5:])
            print(json.dumps(dict(sample=row['sample_id'],stage=record['stage'],progress=progress),ensure_ascii=False))
        return
    if args.action=='package':
        with lock(run/'.lock'):print(package(run,args.release or args.run_id))
        return
    if args.action=='freeze':
        with lock(run/'.lock'):print(freeze(run,args.reports or [],args.representatives or []))
        return
    if args.action=='extend':
        with lock(run/'.lock'):print(extend_run(run,args.count,args.reason))
        return
    if args.limit<1 or args.budget_minutes<=0:raise ValueError('Positive sample limit and budget required')
    speed=args.speed_mm_s if args.speed_mm_s is not None else cfg['loading']['mean_speed_mm_s']
    if args.action in ('run','resume') and manifest['purpose']!='calibration':
        material=audit_material(cfg)
        if not material['passed']:raise RuntimeError('Material audit failed: '+'; '.join(material['failures']))
        accept=read(run/'protocol_acceptance.json') if (run/'protocol_acceptance.json').exists() else {}
        if accept.get('status')!='ACCEPTED':raise RuntimeError('Complete and freeze calibration before a production solve')
        if speed!=accept['mean_speed_mm_s']:raise RuntimeError('Speed differs from accepted protocol')
        if accept.get('config_sha256')!=manifest['config_sha256']:raise RuntimeError('Accepted protocol configuration mismatch')
    start=time.monotonic()
    def remaining():
        value=args.budget_minutes-(time.monotonic()-start)/60
        if value<=0:raise TimeoutError('Budget reached; no further stage submitted')
        return value
    failures=[]
    valid_groups=set()
    for row in manifest['samples']:
        path=run/'samples'/row['sample_id']/'quality.json'
        if path.exists():
            quality=read(path)
            if quality['passed'] and quality.get('geometry_group'):valid_groups.add(quality['geometry_group'])
    with lock(run/'.lock'),runtime.mesh_session(run,cfg,enabled=args.action=='prepare' and not args.fresh_mesher):
        if args.action=='prepare' and args.workers>1:
            from src.parallel_prepare import prepare_parallel
            failures=prepare_parallel(run,cfg,manifest,rows[:args.limit],args,code_hashes,remaining)
            if failures:
                print(json.dumps({'failed_samples':failures}));raise SystemExit(1)
            print('Finished parallel prepare'+(' (Abaqus datacheck off)' if not args.with_datacheck else ''),flush=True)
            return
        if args.action=='run' and not args.samples:
            rows=[r for r in rows if not read(run/'samples'/r['sample_id']/'state.json')['stage'].startswith('CLEANED')]
        for row in rows[:args.limit]:
            if args.action=='run' and manifest['purpose']=='dataset' and len(valid_groups)>=cfg['samples']:
                print('Target valid sample count reached');break
            case=run/'samples'/row['sample_id'];print(row['sample_id']+' '+args.action,flush=True)
            design=read(case/'design.json');declared=design.pop('design_hash')
            if declared!=row['design_hash'] or object_hash(design)!=declared:raise RuntimeError('Design controls changed')
            if time.monotonic()-start>=args.budget_minutes*60:print('Budget reached; no further samples submitted');break
            save(case/'stage_code'/f'{args.action}.json',dict(utc=now(),hashes=code_hashes))
            try:
                if args.action in ('generate','mesh','prepare','run'):
                    with lock(ROOT/'runs/.external.lock'):
                        runtime.assert_external_idle();generate(case,cfg,run/'protocol/source')
                if args.action in ('mesh','prepare','run'):
                    runtime.mesh_case(case,cfg,remaining());validate_mesh(case,cfg)
                    if manifest['purpose']=='dataset' and not args.keep_intermediates:compact_case(case)
                if args.action in ('prepare','run'):
                    deck=write_deck(case,cfg,speed)
                    if args.action=='run' or args.with_datacheck:
                        runtime.datacheck(case,cfg,remaining())
                    if args.action=='prepare':
                        save(case/'preparation.json',dict(utc=now(),input_sha256=deck['input_sha256'],
                             datacheck_requested=args.with_datacheck,
                             scope='Geometry, mesh quality and input preparation; datacheck is optional'))
                    if manifest['purpose']=='dataset' and not args.keep_intermediates:compact_case(case)
                if args.action=='run':runtime.solve(case,cfg,remaining())
                if args.action=='resume':runtime.resume(case,cfg,remaining())
                if args.action in ('extract','run','resume'):
                    runtime.extract(case,cfg,remaining());quality=postprocess(case,cfg);runtime.cleanup(case)
                    if quality['passed'] and quality.get('geometry_group'):valid_groups.add(quality['geometry_group'])
                if args.action=='diagnose':
                    runtime.extract(case,cfg,remaining());runtime.archive_failure(case)
                if args.action=='cleanup':runtime.cleanup(case)
                if args.action=='compact':print(json.dumps(compact_case(case)),flush=True)
            except Exception as exc:
                save(case/'last_error.json',dict(action=args.action,error=str(exc),utc=now()))
                if args.action in ('generate','mesh','prepare','run') and manifest['purpose']=='dataset' and not args.keep_intermediates:
                    try:compact_case(case)
                    except Exception as storage_error:
                        save(case/'storage_error.json',dict(error=str(storage_error),utc=now()))
                sta=case/'analysis.sta'
                if sta.exists() and not (case/'analysis.lck').exists() and (case/'analysis.odb').exists():
                    if 'THE ANALYSIS HAS NOT BEEN COMPLETED' in sta.read_text(errors='replace'):
                        try:
                            runtime.extract(case,cfg,remaining());runtime.archive_failure(case)
                        except Exception as diagnostic_error:
                            save(case/'diagnostic_error.json',dict(error=str(diagnostic_error),utc=now()))
                failures.append(dict(sample_id=row['sample_id'],error=str(exc)))
                if not args.continue_on_error or isinstance(exc,TimeoutError) or runtime.active_external():raise
                print('REJECTED '+row['sample_id']+': '+str(exc),flush=True)
    if failures:
        print(json.dumps({'failed_samples':failures}));raise SystemExit(1)
    suffix=' (Abaqus datacheck off)' if args.action=='prepare' and not args.with_datacheck else ''
    print('Finished '+args.action+suffix,flush=True)


if __name__=='__main__':
    main()
