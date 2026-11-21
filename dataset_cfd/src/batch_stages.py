"""Two resumable, independent batch stages around the existing case workers."""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from collections import Counter
import psutil
from .common import ROOT,read,save,digest,lock,now
from .legacy_protocol import validate_cached_mesh_variant
from .trimatic_session import TrimaticSession

DEFAULT_PROFILE=ROOT/'configs/legacy_adaptive_surface015_candidate.json'


def run_root(name):
    if not name or not name.replace('_','').replace('-','').isalnum():
        raise ValueError('Run name must contain only letters, digits, underscores and hyphens')
    return ROOT/'runs'/name


def select_samples(inventory, samples=None, all_samples=False, pilot=False):
    known={r['sample_id']:r for r in inventory['samples']}
    ids=(list(known) if all_samples else inventory['pilot_selection']['sample_ids'] if pilot else samples)
    if not ids:raise ValueError('Choose --all, --pilot or --samples')
    if len(set(ids))!=len(ids) or not set(ids)<=set(known):
        raise ValueError('Sample IDs must be unique and present in the source inventory')
    return list(ids)


def _sources(stage):
    worker='legacy_pilot.py' if stage=='geometry' else 'legacy_solve_cached.py'
    return [*sorted((ROOT/'src').glob('*.py')),ROOT/f'{1 if stage=="geometry" else 2}_batch_{"fluid" if stage=="geometry" else "cfd"}.py',
            ROOT/'workers'/worker,ROOT/'workers/watch_legacy_trial.py',
            *([ROOT/'workers/legacy_fluid_blender.py',ROOT/'workers/legacy_remesh_3matic.py',
               ROOT/'workers/legacy_3matic_session.py']
              if stage=='geometry' else [ROOT/'workers/verify_legacy_physics.py'])]


def _code_hashes(stage):
    return {p.relative_to(ROOT).as_posix():digest(p) for p in _sources(stage)}


def prepare_run(name, stage, profile=None, inventory_source=None):
    root=run_root(name)
    inventory_source=Path(inventory_source) if inventory_source else ROOT/'plan/source_inventory_20261107.json'
    with lock(ROOT/'.batch_initialization.lock'):
        if (root/'run.json').exists():
            meta=read(root/'run.json')
            if digest(root/'protocol.json')!=meta['protocol_sha256']:
                raise RuntimeError('Frozen profile changed; use a new run name')
            if profile and read(profile)!=read(root/'protocol.json'):
                raise RuntimeError('Requested profile differs; use a new run name')
            if digest(inventory_source)!=meta['inventory_sha256']:
                raise RuntimeError('Source inventory changed; use a new run name')
        else:
            root.mkdir(parents=True,exist_ok=True)
            if (root/'protocol.json').exists():
                raise RuntimeError('Incomplete run initialization retained; use a new run name')
            shutil.copy2(profile or DEFAULT_PROFILE,root/'protocol.json')
            shutil.copy2(inventory_source,root/'source_inventory.json')
            meta={'created_utc':now(),'protocol_sha256':digest(root/'protocol.json'),
                  'inventory_sha256':digest(root/'source_inventory.json'),
                  'policy':'failed cases retained and skipped; retries explicit; no automatic replacement',
                  'production_protocol_frozen':False}
            save(root/'run.json',meta)
        if digest(root/'source_inventory.json')!=meta['inventory_sha256']:
            raise RuntimeError('Frozen source inventory changed')
        snapshot=root/f'{stage}_source'
        current=_code_hashes(stage)
        if (snapshot/'manifest.json').exists():
            if read(snapshot/'manifest.json')!=current:
                raise RuntimeError('Stage code changed; use a new run name (CFD may reuse --geometry-run)')
        else:
            snapshot.mkdir(exist_ok=False)
            for p in _sources(stage):
                target=snapshot/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(p,target)
            save(snapshot/'manifest.json',current)
    return root,meta,current


def _process(record, scripts):
    if not record or 'created' not in record:return None
    try:
        p=psutil.Process(record['pid'])
        if abs(p.create_time()-record['created'])>.1:return None
        command=p.cmdline()
        if not any(Path(arg).name in scripts for arg in command):
            raise RuntimeError('Live receipt belongs to an unexpected command; no process was touched')
        return p
    except psutil.NoSuchProcess:return None


def live_case(case):
    for name,scripts in [('process.json',{'legacy_pilot.py','legacy_solve_cached.py'}),
                         ('native_process.json',{'legacy_fluid_blender.py','legacy_remesh_3matic.py'})]:
        if (case/name).exists():
            r=read(case/name)
            if r.get('status')=='RUNNING' and _process(r,scripts):return True
    return False


def _stop_case(case, child, created, reason):
    """Stop only this batch's verified worker and its recorded native child."""
    stopped=[]
    native=case/'native_process.json'
    records=[]
    if native.exists():
        r=read(native)
        if r.get('status')=='RUNNING':records.append((r,{'legacy_fluid_blender.py','legacy_remesh_3matic.py'}))
    records.append(({'pid':child.pid,'created':created},{'legacy_pilot.py','legacy_solve_cached.py'}))
    for receipt,scripts in records:
        p=_process(receipt,scripts)
        if p:
            p.terminate()
            try:p.wait(timeout=10)
            except psutil.TimeoutExpired:
                p=_process(receipt,scripts)
                if p:p.kill();p.wait(timeout=10)
            stopped.append(receipt['pid'])
    save(case/'batch_interruption.json',{'utc':now(),'reason':reason,'stopped_owned_pids':stopped,
                                        'checkpoints_retained':True})


def execute_case(command, case, log_prefix, timeout_s, on_start, extra_env=None):
    env=os.environ.copy();env['PYTHONIOENCODING']='utf-8'
    if extra_env:env.update(extra_env)
    flags=getattr(subprocess,'CREATE_NO_WINDOW',0)
    started=time.monotonic();watcher=None
    with (log_prefix.with_suffix('.stdout.log').open('wb') as out,
          log_prefix.with_suffix('.stderr.log').open('wb') as err,
          log_prefix.with_suffix('.watch.log').open('wb') as watch_log):
        child=subprocess.Popen(command,cwd=ROOT.parent,env=env,stdout=out,stderr=err,creationflags=flags)
        created=psutil.Process(child.pid).create_time();on_start(child.pid,created)
        try:
            while child.poll() is None:
                if watcher is None and (case/'process.json').exists():
                    owner=read(case/'process.json')
                    if owner['pid']!=child.pid or abs(owner['created']-created)>.1:
                        raise RuntimeError('Case receipt differs from launched worker')
                    watcher=subprocess.Popen([sys.executable,str(ROOT/'workers/watch_legacy_trial.py'),
                        '--case',str(case)],cwd=ROOT.parent,env=env,stdout=watch_log,stderr=watch_log,creationflags=flags)
                if watcher is not None and watcher.poll() not in (None,0):
                    raise RuntimeError('Resource monitor failed')
                if time.monotonic()-started>timeout_s:
                    _stop_case(case,child,created,'CASE_TIMEOUT');return -100
                time.sleep(1)
        except BaseException:
            _stop_case(case,child,created,'BATCH_INTERRUPTED');raise
    if live_case(case):
        _stop_case(case,child,created,'NATIVE_TIMEOUT_RETAINED');return -101
    return child.returncode


def validate_case(case, stage, protocol_hash, inventory_row, source_geometry=None):
    """Read actual receipts/hashes; output-file existence alone is insufficient."""
    if live_case(case):raise RuntimeError('Case still has a live owner')
    p=read(case/'process.json');source=read(case/'source.json')
    if p['status']!='FINISHED':raise ValueError('Case worker did not finish successfully')
    if source['sample_id']!=inventory_row['sample_id'] or source['source_sample_id']!=inventory_row['source_sample_id']:
        raise ValueError('Case identity differs from source inventory')
    if source['split']!=inventory_row['split'] or source['geometry_group']!=inventory_row['geometry_group']:
        raise ValueError('Split or geometry-group provenance differs')
    source_file = 'remeshed.stl' if source.get('solid_source') == 'remeshed_stl' else 'analysis.inp'
    if digest(inventory_row['files'][source_file]['path'])!=source['source_inp_sha256']:
        raise ValueError('Initial solid source changed')
    if digest(case/'protocol.json')!=protocol_hash or p['protocol_sha256']!=protocol_hash:
        raise ValueError('Case protocol differs from the frozen run')
    g=read(case/'fluid_report.json')
    if (case/'shared_native_request.json').exists():
        if read(case/'shared_native_request.json')['status']!='FINISHED':
            raise ValueError('Shared native request did not complete')
    if not g['accepted_for_development'] or g['sha256']!=digest(case/'fluid_domain.stl'):
        raise ValueError('Fluid surface failed validation or hash check')
    if g['config_sha256']!=protocol_hash:raise ValueError('Geometry protocol hash differs')
    if stage=='cfd':
        r=read(case/'results.json');audit=read(case/'physics_verification.json')
        if not r['cfd_valid'] or not r['flow_solution_available'] or not audit['passed']:
            raise ValueError('Flow or saved-physics checks failed')
        if r['fluid_sha256']!=g['sha256'] or r['protocol_sha256']!=protocol_hash:
            raise ValueError('Flow solution input hashes differ')
        if source_geometry and g['sha256']!=read(source_geometry/'fluid_report.json')['sha256']:
            raise ValueError('CFD input differs from the selected geometry-stage result')
    return True


def _failure(case):
    process=read(case/'process.json') if (case/'process.json').exists() else {}
    st=read(case/'state.json') if (case/'state.json').exists() else {}
    stages=list(st.get('stages',{}))
    active=next((x for x in reversed(stages) if not x.endswith('FAILED')),st.get('stage'))
    error=process.get('error','Worker returned without a verified result')
    return {'failure_stage':active,'error':error,'error_last_line':error.strip().splitlines()[-1]}


def inherit_stage(source_root, stage, meta, rows):
    """Index completed old cases without copying, modifying or rerunning them."""
    source_root=Path(source_root);origin=read(source_root/'run.json')
    if origin['protocol_sha256']!=meta['protocol_sha256'] or origin['inventory_sha256']!=meta['inventory_sha256']:
        raise ValueError('Inherited run uses another protocol or source inventory')
    manifest=source_root/f'batch_{stage}.json';old=read(manifest)
    if _process(old,{'1_batch_fluid.py','2_batch_cfd.py'}):
        raise RuntimeError('Inherited stage still has a live coordinator')
    data={'stage':stage,'created_utc':now(),'jobs':old['jobs'],'invocations':[],
          'inheritance':{'run':str(source_root),'manifest_sha256':digest(manifest),
                         'run_metadata_sha256':digest(source_root/'run.json'),
                         'source_manifest_sha256':digest(source_root/f'{stage}_source/manifest.json')}}
    for sid,job in data['jobs'].items():
        job['inherited_from_run']=str(source_root)
        if not job.get('attempts'):continue
        case=Path(job['attempts'][-1]['case'])
        if live_case(case):raise RuntimeError('Inherited case is still live')
        if job['status']=='SUCCESS':validate_case(case,stage,meta['protocol_sha256'],rows[int(sid)])
        elif job['status'] in ('STARTING','RUNNING'):
            try:
                validate_case(case,stage,meta['protocol_sha256'],rows[int(sid)])
                job.update(status='SUCCESS',case=str(case));job['attempts'][-1]['status']='SUCCESS'
            except (ValueError,FileNotFoundError):
                p=read(case/'process.json') if (case/'process.json').exists() else {}
                if p.get('status') in ('FAILED','RESOURCE_STOP_RETAINED'):
                    job.update(status='FAILED',**_failure(case));job['attempts'][-1]['status']='FAILED'
                else:job['status']='INTERRUPTED'
    return data


def stage_cli(stage, argv=None):
    parser=argparse.ArgumentParser(description=f'Batch {stage}: continue past failed cases and resume completed work')
    parser.add_argument('--run',required=True)
    parser.add_argument('--profile',type=Path)
    parser.add_argument('--inventory',type=Path,help='Campaign source inventory (default: plan/source_inventory_20261107.json)')
    parser.add_argument('--inherit-run',help='Preserve checked cases from a stopped run with identical inputs/settings')
    choose=parser.add_mutually_exclusive_group()
    choose.add_argument('--all',action='store_true');choose.add_argument('--pilot',action='store_true')
    choose.add_argument('--samples',type=int,nargs='+')
    parser.add_argument('--status',action='store_true')
    parser.add_argument('--retry-failed',action='store_true')
    parser.add_argument('--max-attempts',type=int,default=2)
    parser.add_argument('--max-cases',type=int)
    parser.add_argument('--budget-minutes',type=float)
    parser.add_argument('--case-timeout-minutes',type=float,default=20 if stage=='geometry' else 60)
    if stage=='cfd':parser.add_argument('--geometry-run',help='Reuse geometry from another staged run')
    if stage=='geometry':
        parser.add_argument('--trimatic-mode',choices=['persistent','isolated'],default='persistent',
                            help='One 3-matic session per batch (default), or per-case isolated debugging')
    args=parser.parse_args(argv)
    if args.status:
        data=read(run_root(args.run)/f'batch_{stage}.json')
        print(json.dumps({k:v for k,v in data.items() if k!='jobs'},ensure_ascii=False,indent=2));return 0
    if (args.max_attempts<1 or (args.max_cases is not None and args.max_cases<1)
            or not math.isfinite(args.case_timeout_minutes) or args.case_timeout_minutes<=0):
        parser.error('Attempt/case limits and timeout must be positive')
    if args.budget_minutes is not None and (not math.isfinite(args.budget_minutes) or args.budget_minutes<=0):
        parser.error('Budget must be finite and positive')
    # Validate selection before creating a frozen run.
    inventory=read(args.inventory or ROOT/'plan/source_inventory_20261107.json')
    ids=select_samples(inventory,args.samples,args.all,args.pilot)
    with lock(ROOT/f'.{stage}_batch.lock'):
        root,meta,hashes=prepare_run(args.run,stage,args.profile,args.inventory)
        rows={r['sample_id']:r for r in read(root/'source_inventory.json')['samples']}
        manifest=root/f'batch_{stage}.json'
        if manifest.exists():
            data=read(manifest)
            if args.inherit_run and data.get('inheritance',{}).get('run')!=str(run_root(args.inherit_run)):
                raise RuntimeError('Inherited source changed; create a new run')
        elif args.inherit_run:data=inherit_stage(run_root(args.inherit_run),stage,meta,rows)
        else:data={'stage':stage,'created_utc':now(),'jobs':{},'invocations':[]}
        if stage=='geometry':
            if data.get('trimatic_mode',args.trimatic_mode)!=args.trimatic_mode:
                raise RuntimeError('3-matic execution mode changed; create a new run')
            data['trimatic_mode']=args.trimatic_mode
        geometry_root=run_root(getattr(args,'geometry_run',None) or args.run)
        if stage=='cfd':
            if data.get('geometry_run',str(geometry_root))!=str(geometry_root):
                raise RuntimeError('Geometry source changed; use a new CFD run name')
            data['geometry_run']=str(geometry_root)
        data.update(status='RUNNING',pid=os.getpid(),created=psutil.Process().create_time(),active_sample=None,
                    updated_utc=now(),protocol_sha256=meta['protocol_sha256'])
        data.pop('finished_utc',None);data.pop('error',None)
        data['invocations'].append({'utc':now(),'sample_ids':ids,'retry_failed':args.retry_failed})
        def persist():
            data.update(updated_utc=now(),counts=dict(Counter(j['status'] for j in data['jobs'].values())))
            save(manifest,data)
        persist();started=time.monotonic();launched=0
        session=TrimaticSession(root) if stage=='geometry' and args.trimatic_mode=='persistent' else None
        try:
            for sid in ids:
                if _code_hashes(stage)!=hashes or digest(root/'protocol.json')!=meta['protocol_sha256']:
                    raise RuntimeError('Code or profile changed during execution; use a new run')
                job=data['jobs'].setdefault(str(sid),{'sample_id':sid,'status':'PENDING','attempts':[]})
                source_case=None
                if stage=='cfd':
                    gm=geometry_root/'batch_geometry.json'
                    gj=read(gm)['jobs'].get(str(sid)) if gm.exists() else None
                    if not gj or gj['status']!='SUCCESS':
                        if job['status']=='SUCCESS':raise RuntimeError('Previously used geometry is no longer available')
                        job.update(status='NO_GEOMETRY',reason='No successful fluid-domain result');persist();continue
                    source_case=Path(gj['case'])
                    validate_case(source_case,'geometry',digest(source_case/'protocol.json'),rows[sid])
                    validate_cached_mesh_variant(read(source_case/'protocol.json'),read(root/'protocol.json'))
                if job['attempts']:
                    last=job['attempts'][-1];previous=Path(last['case'])
                    if live_case(previous):raise RuntimeError(f'Sample {sid} still has a live worker; retained')
                    if job['status'] in ('SUCCESS','RUNNING','STARTING'):
                        try:
                            validate_case(previous,stage,meta['protocol_sha256'],rows[sid],source_case)
                            job.update(status='SUCCESS',case=str(previous));persist();continue
                        except (ValueError,FileNotFoundError):
                            if job['status']=='SUCCESS':
                                raise RuntimeError(f'Previously successful sample {sid} changed; use a new run')
                            job.update(status='INTERRUPTED')
                    if job['status']=='FAILED' and not args.retry_failed:continue
                    if len(job['attempts'])>=args.max_attempts:continue
                if ((args.max_cases is not None and launched>=args.max_cases) or
                    (args.budget_minutes is not None and time.monotonic()-started>=args.budget_minutes*60)):
                    data['status']='BUDGET_REACHED';break
                if session:session.ensure_started()
                attempt=len(job['attempts'])+1;worker_run=f'{args.run}__{stage}_a{attempt:03d}'
                case=run_root(worker_run)/'samples'/str(sid)
                if case.exists():raise RuntimeError(f'Untracked case retained: {case}')
                item={'attempt':attempt,'case':str(case),'status':'STARTING','started_utc':now()}
                job['attempts'].append(item);job.update(status='STARTING',case=str(case))
                data['active_sample']=sid;persist()
                logdir=root/'logs';logdir.mkdir(exist_ok=True)
                prefix=logdir/f'{stage}_{sid}_a{attempt:03d}'
                if stage=='geometry':
                    command=[sys.executable,str(ROOT/'workers/legacy_pilot.py'),'--geometry-only','--sample',str(sid),
                             '--run',worker_run,'--profile',str(root/'protocol.json')]
                else:
                    command=[sys.executable,str(ROOT/'workers/legacy_solve_cached.py'),'--source-case',str(source_case),
                             '--run',worker_run,'--profile',str(root/'protocol.json')]
                def launched_case(pid,created):
                    item.update(pid=pid,created=created,status='RUNNING');job['status']='RUNNING';persist()
                print(f'[{stage}] sample {sid}, attempt {attempt}',flush=True)
                code=execute_case(command,case,prefix,args.case_timeout_minutes*60,launched_case,
                    {'CFD_V2_INVENTORY':str(root/'source_inventory.json')});launched+=1
                if session:session.recover_after_case(case)
                if _code_hashes(stage)!=hashes or digest(root/'protocol.json')!=meta['protocol_sha256']:
                    raise RuntimeError('Code or profile changed while the case ran; create a new run')
                if stage=='cfd' and code==0:
                    with prefix.with_suffix('.verification.log').open('wb') as log:
                        check=subprocess.run([sys.executable,str(ROOT/'workers/verify_legacy_physics.py'),'--case',str(case)],
                            cwd=ROOT.parent,stdout=log,stderr=log,timeout=90,
                            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                    code=check.returncode
                item.update(exit_code=code,finished_utc=now())
                try:
                    if code!=0:raise ValueError(f'Worker exit code {code}')
                    validate_case(case,stage,meta['protocol_sha256'],rows[sid],source_case)
                    job['status']='SUCCESS';item['status']='SUCCESS'
                except (ValueError,FileNotFoundError) as error:
                    job.update(status='FAILED',**_failure(case));item.update(status='FAILED',validation_error=str(error))
                    if 'Another CFD writer holds this lock' in job['error']:
                        job['status']='INTERRUPTED';raise RuntimeError('Another worker owns a required lock; resume later')
                persist();print(f'[{stage}] sample {sid}: {job["status"]}',flush=True)
            else:data['status']='FINISHED'
            data.update(active_sample=None,finished_utc=now());persist()
        except BaseException:
            data.update(status='STOPPED_RETAINED',error=traceback.format_exc());persist();raise
        finally:
            if session:session.close()
        print(json.dumps({'stage':stage,'status':data['status'],'counts':data['counts'],'manifest':str(manifest)},ensure_ascii=False))
    return 0
