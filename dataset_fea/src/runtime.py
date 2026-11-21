from __future__ import annotations
import gzip
import copy
from contextlib import contextmanager
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
import zipfile
import psutil
from .common import ROOT,inside,save,read,digest,verify_files,state,now,lock


def hidden_options():
    if os.name!='nt':return {}
    info=subprocess.STARTUPINFO();info.dwFlags|=subprocess.STARTF_USESHOWWINDOW;info.wShowWindow=0
    return dict(startupinfo=info,creationflags=subprocess.CREATE_NO_WINDOW)


REGISTRY=ROOT/'runs/external_process.json'
WORKER_SOURCE=ROOT/'workers'
SCRATCH_EXTENSIONS={'.odb','.abq','.res','.sel','.stt','.mdl','.pac','.prt','.com','.sim'}
_MESH_SESSION=None
_STALL_MINUTES=None
_INITIALIZATION_MINUTES=None


class StageStalledError(RuntimeError):
    pass


class InitializationStalledError(StageStalledError):
    pass


@contextmanager
def batch_execution(stall_minutes=30,initialization_minutes=5):
    """No total deadline; detect an external stage that stops producing output."""
    global _STALL_MINUTES,_INITIALIZATION_MINUTES
    previous=(_STALL_MINUTES,_INITIALIZATION_MINUTES)
    _STALL_MINUTES=stall_minutes;_INITIALIZATION_MINUTES=initialization_minutes
    try:yield
    finally:_STALL_MINUTES,_INITIALIZATION_MINUTES=previous


def explicit_stage_state(case,log_name,job_name):
    """Identify solver startup separately from input processing and increment output."""
    def tail(path):
        try:
            with path.open('rb') as stream:
                stream.seek(0,2);size=stream.tell();stream.seek(max(0,size-131072))
                return stream.read().decode(errors='replace')
        except FileNotFoundError:return ''
    status=tail(case/(job_name+'.sta'))
    if re.search(r'^\s*\d+\s+[-+\d.]+[Ee][-+]?\d+\s+[-+\d.]+[Ee][-+]?\d+',status,re.M):
        return 'initialized'
    if re.search(r'Run explicit(?:_dp)?\.exe',tail(case/log_name)):
        return 'initializing'
    return 'preprocessing'


def stage_progress(case):
    progress=[]
    for path in case.iterdir():
        if path.is_file() and (path.name.startswith(('analysis.','preflight.','raw_history.','energy_history.'))
                              or path.suffix in ('.log','.tmp')):
            info=path.stat();progress.append((path.name,info.st_size,info.st_mtime_ns))
    return tuple(sorted(progress))


@contextmanager
def mesh_session(run,cfg,enabled=True):
    """Lazily start one reusable mesher for this preparation invocation."""
    global _MESH_SESSION
    if not enabled:
        yield
        return
    if _MESH_SESSION is not None:raise RuntimeError('Mesher session already configured')
    from .mesh_session import MeshSession
    session=MeshSession(run,cfg);_MESH_SESSION=session
    try:yield session
    finally:
        try:session.close()
        finally:_MESH_SESSION=None


def active_external():
    """Persistent process guard survives the Python coordinator exiting or timing out."""
    if not REGISTRY.exists():return None
    record=read(REGISTRY)
    if record.get('status')!='RUNNING':return None
    # Only the owning coordinator may use disk artifacts while its own mesher is idle.
    # Other coordinators still see the resident process and cannot launch over it.
    owner=record.get('owner',{})
    if (record.get('kind')=='meshing_session' and not record.get('busy',True)
            and owner.get('pid')==os.getpid()
            and abs(owner.get('created',0)-psutil.Process().create_time())<.01):
        return None
    live=[]
    for identity in record.get('processes',[]):
        try:
            proc=psutil.Process(identity['pid'])
            if abs(proc.create_time()-identity['created'])<.01 and proc.is_running():live.append(identity)
        except psutil.NoSuchProcess:pass
    locks=list(inside(record['case']).glob('*.lck'))
    if live or locks:return dict(record,live=live,locks=[str(p) for p in locks])
    save(REGISTRY,dict(record,status='ENDED',checked_utc=now()))
    return None


def assert_external_idle():
    record=active_external()
    if record:raise RuntimeError('External process remains active in '+record['case']+'; inspect status or resume that exact job')


def recovery_processes(case,record):
    """Check saved identities and Abaqus children orphaned by an early Ctrl+C."""
    live={}
    for identity in record.get('processes',[]):
        try:
            process=psutil.Process(identity['pid'])
            if abs(process.create_time()-identity['created'])<.01 and process.is_running():
                live[process.pid]=dict(pid=process.pid,created=process.create_time())
        except psutil.NoSuchProcess:pass
    for process in psutil.process_iter(['pid','name','cmdline']):
        try:
            name=(process.info['name'] or '').lower()
            if not (name.startswith(('abq','explicit','standard','sma')) or
                    name in ('pre.exe','package.exe','mpiexec.exe')):continue
            command=' '.join(process.info['cmdline'] or []).lower().replace('\\','/')
            belongs=str(case).lower().replace('\\','/') in command or Path(process.cwd()).resolve()==case
            if belongs:live[process.pid]=dict(pid=process.pid,created=process.create_time())
        except psutil.NoSuchProcess:pass
        except psutil.AccessDenied as exc:
            raise RuntimeError('Cannot verify Abaqus process ownership before interrupted-job recovery') from exc
    return list(live.values())


def recover_interrupted_run(run):
    """Release a dead, tracked job; retain live suspended jobs and complete results."""
    run=inside(run)
    if not REGISTRY.exists():return None
    record=read(REGISTRY)
    if record.get('status') not in ('RUNNING','ENDED'):return None
    case=Path(record.get('case','')).resolve()
    if case.parent!=(run/'samples').resolve():return None
    command=record.get('command',[])
    job=next((arg[4:] for arg in command if arg.startswith('job=')),None)
    if job not in ('preflight','analysis') or not record.get('processes'):return None
    with lock(ROOT/'runs/.external.lock'):
        if recovery_processes(case,record):return None
        verify_files(case,{'analysis.inp':read(case/'analysis.json')['input_sha256']})
        if (case/'cleanup_receipt.json').exists() or (case/'export_manifest.json').exists():
            if not (case/'export_manifest.json').exists():
                raise RuntimeError('Interrupted job has an incomplete cleanup receipt; preserve results')
            verify_files(case,read(case/'export_manifest.json')['hashes'])
            complete=True
        else:
            sta=case/(job+'.sta')
            complete=job=='analysis' and sta.exists() and 'THE ANALYSIS HAS COMPLETED SUCCESSFULLY' in sta.read_text(errors='replace')
        check=read(case/'datacheck.json') if (case/'datacheck.json').exists() else None
        if job=='preflight' and check and check.get('input_sha256')!=digest(case/'analysis.inp'):
            raise RuntimeError('Interrupted datacheck input changed; preserve artifacts')
        preserved=complete or job=='preflight' and check and check.get('passed')
        scratch=[]
        for path in case.iterdir():
            if not path.is_file():continue
            if path.name==job+'.lck' or (not preserved and path.name.startswith(job+'.') and
                    path.name not in ('analysis.inp','analysis.json')):
                scratch.append(path)
        partial_results=('extraction.json','raw_history.csv.gz','energy_history.json.gz','field_diagnostics.json',
                         'sparse_displacements.json.gz','metrics.json','quality.json','curve.npz','curve.png','stress_strain.csv',
                         'failed_export_manifest.json','failure_diagnostics.json')
        if job=='analysis' and not preserved:
            scratch.extend(case/name for name in partial_results if (case/name).exists())
        if job=='preflight' and check and not preserved:scratch.append(case/'datacheck.json')
        audit=ROOT/'audits/runtime/interrupted_recovery'/run.name/case.name/str(time.time_ns())
        audit.mkdir(parents=True)
        save(audit/'before.json',dict(registry=record,state=read(case/'state.json')))
        with zipfile.ZipFile(audit/'logs.zip','w',zipfile.ZIP_DEFLATED) as archive:
            for path in case.iterdir():
                if path.is_file() and (path.suffix in ('.sta','.dat','.msg','.log','.lck','.json') or '.msg.' in path.name):
                    archive.write(path,path.name)
        # Check again immediately before removing files; another coordinator cannot
        # launch while this OS lock is held.
        if recovery_processes(case,record):raise RuntimeError('Interrupted job became active during recovery')
        deleted=[]
        for path in scratch:
            path=inside(path,case)
            if path.exists():deleted.append(dict(name=path.name,bytes=path.stat().st_size));path.unlink()
        result=dict(utc=now(),case=str(case),job=job,deleted=deleted,complete_result_preserved=bool(complete),
                    action='REUSE_COMPLETE' if complete else 'RESTART_INCOMPLETE',audit=str(audit))
        save(audit/'recovery.json',result)
        save(case/'interruption_recovery.json',result)
        state(case,'SOLVED' if complete else 'DATACHECK_OK' if check and check.get('passed') else 'PREPARED',
              reason='Automatically recovered an interrupted job after verifying all owned processes exited',audit=str(audit))
        save(REGISTRY,dict(record,status='RECOVERED_INTERRUPTION',checked_utc=now(),audit=str(audit)))
        print(case.name+': recovered interrupted '+job+'; '+result['action'].lower(),flush=True)
        return result


def stage_minutes(cfg,budget_minutes):
    """Honor an explicit caller budget, including unlimited production stages."""
    return cfg['runtime']['max_stage_minutes'] if budget_minutes is None else budget_minutes


def launch(command,case,log_name,minutes,env=None,solver_job=None):
    """Bounded subprocess; never kill a solver tree by image name."""
    case=inside(case)
    if minutes<=0:raise TimeoutError('Budget exhausted before external stage')
    # Abaqus must not overwrite the registry while an idle resident mesher exists.
    if _MESH_SESSION is not None:_MESH_SESSION.close(locked=True)
    assert_external_idle()
    started=time.monotonic()
    with (case/log_name).open('w',encoding='utf-8') as log:
        process=subprocess.Popen([str(x) for x in command],cwd=case,stdout=log,stderr=subprocess.STDOUT,
                                 env=env,**hidden_options())
        save(case/'process.json',dict(pid=process.pid,command=[str(x) for x in command],started_utc=now()))
        tracked={}
        def register():
            try:
                parent=psutil.Process(process.pid)
                for proc in [parent]+parent.children(recursive=True):
                    try:tracked[proc.pid]=dict(pid=proc.pid,created=proc.create_time())
                    except psutil.NoSuchProcess:pass
            except psutil.NoSuchProcess:pass
            save(REGISTRY,dict(status='RUNNING',case=str(case),command=[str(x) for x in command],
                               solver_job=solver_job,processes=list(tracked.values()),updated_utc=now()))
        register();next_register=started+10
        last_progress=started;signature=stage_progress(case);next_progress_check=started
        job_name=solver_job or next((str(x)[4:] for x in command if str(x).startswith('job=')),None)
        initialization_started=None;initialized=False
        try:
            while process.poll() is None:
                if time.monotonic()>=next_register:register();next_register=time.monotonic()+10
                if _STALL_MINUTES is not None and time.monotonic()>=next_progress_check:
                    current=stage_progress(case)
                    if current!=signature:signature=current;last_progress=time.monotonic()
                    if _INITIALIZATION_MINUTES is not None and job_name and not initialized:
                        phase=explicit_stage_state(case,log_name,job_name)
                        if phase=='initialized':initialized=True
                        elif phase=='initializing':
                            if initialization_started is None:initialization_started=time.monotonic()
                            if time.monotonic()-initialization_started>=_INITIALIZATION_MINUTES*60:
                                register()
                                raise InitializationStalledError('Explicit initialization has not reached increment zero for %g minutes in %s' %
                                                        (_INITIALIZATION_MINUTES,log_name))
                    if time.monotonic()-last_progress>_STALL_MINUTES*60:
                        register()
                        raise StageStalledError('No output progress for %g minutes in %s' % (_STALL_MINUTES,log_name))
                    next_progress_check=time.monotonic()+10
                if time.monotonic()-started>minutes*60:
                    if solver_job:
                        register()
                        subprocess.run([str(command[0]),'suspend','job='+solver_job],cwd=case,
                                       stdout=log,stderr=subprocess.STDOUT,timeout=45,**hidden_options())
                        state(case,'BUDGET_SUSPEND_REQUESTED',job=solver_job)
                        raise TimeoutError('Budget reached; requested Abaqus suspend. Check status before resume.')
                    # For meshing, preserve an active external process and never launch another on top.
                    state(case,'EXTERNAL_PROCESS_RUNNING',pid=process.pid)
                    register()
                    raise TimeoutError('Stage budget reached; external process preserved; inspect receipt/status')
                time.sleep(1)
        except KeyboardInterrupt:
            register()
            if solver_job:
                subprocess.run([str(command[0]),'suspend','job='+solver_job],cwd=case,
                               stdout=log,stderr=subprocess.STDOUT,timeout=45,**hidden_options())
                state(case,'USER_SUSPEND_REQUESTED',job=solver_job)
            raise
    elapsed=time.monotonic()-started
    # Keep child processes or a solver lock registered if the launcher exited early.
    active_external()
    save(case/(log_name+'.receipt.json'),dict(command=[str(x) for x in command],returncode=process.returncode,elapsed_s=elapsed))
    if process.returncode:raise RuntimeError('External command failed; inspect '+str(case/log_name))
    return elapsed


def stop_case(case,cfg):
    """Release only this case's registered process identities before continuing."""
    case=inside(case);record=active_external()
    if record and ('case' not in record or Path(record['case']).resolve()!=case.resolve()):
        raise RuntimeError('An external process belongs to another case; cannot stop it')
    if record:
        identities={i['pid']:i for i in record.get('processes',[])}
        for identity in list(identities.values()):
            try:
                process=psutil.Process(identity['pid'])
                if abs(process.create_time()-identity['created'])<.01:
                    for child in process.children(recursive=True):identities[child.pid]=dict(pid=child.pid,created=child.create_time())
            except psutil.NoSuchProcess:pass
        job=next((arg.split('=',1)[1] for arg in record.get('command',[]) if arg.startswith('job=')),None)
        if job:
            with (case/'automatic_stop.log').open('a') as log:
                try:subprocess.run([os.environ.get("LOGIC_ABAQUS", cfg['runtime']['abaqus']),'terminate','job='+job],cwd=case,
                        stdout=log,stderr=subprocess.STDOUT,timeout=15,**hidden_options())
                except (subprocess.TimeoutExpired,OSError):pass
        def live():
            processes=[]
            for identity in identities.values():
                try:
                    process=psutil.Process(identity['pid'])
                    if abs(process.create_time()-identity['created'])<.01 and process.is_running():processes.append(process)
                except psutil.NoSuchProcess:pass
            return processes
        if job:psutil.wait_procs(live(),timeout=5)
        for process in live():
            try:
                if abs(process.create_time()-identities[process.pid]['created'])<.01:process.kill()
            except psutil.NoSuchProcess:pass
        psutil.wait_procs(live(),timeout=5)
        if live():raise RuntimeError('Failed case still has live processes')
        for path in case.glob('*.lck'):path.unlink(missing_ok=True)
        save(REGISTRY,dict(record,status='STOPPED_FAILED_CASE',checked_utc=now()))
    if list(case.glob('*.lck')):raise RuntimeError('Untracked lock remains in failed case')
    assert_external_idle()


def mesh_case(case,cfg,budget_minutes=None):
    if (case/'mesher_result.json').exists():
        result=read(case/'mesher_result.json')
        if result['status']!='MESH_EXPORTED':raise RuntimeError('Prior meshing failure preserved; inspect mesher_result.json')
        return
    if (case/'mesher_config.json').exists():raise RuntimeError('Meshing was already started; inspect external process/receipt')
    save(case/'mesher_config.json',cfg['mesh'])
    env=os.environ.copy();env['FEA_V2_CASE']=str(case.resolve())
    with lock(ROOT/'runs/.external.lock'):
        state(case,'MESHING')
        minutes=stage_minutes(cfg,budget_minutes)
        if _MESH_SESSION is not None:_MESH_SESSION.mesh(case,minutes)
        else:
            launch([os.environ.get("LOGIC_TRIMATIC", cfg['runtime']['trimatic']),'-run_script',WORKER_SOURCE/'mesh_3matic.py'],case,'mesher_console.log',minutes,env=env)


def clean_preflight(case):
    case=inside(case)
    if (case/'preflight.lck').exists():raise RuntimeError('Datacheck is active')
    if not read(case/'datacheck.json')['passed']:return
    archive=case/'datacheck_logs.zip'
    if not archive.exists():
        temp=archive.with_suffix('.zip.tmp')
        with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED) as z:
            for name in ('preflight.sta','preflight.dat','preflight.msg','preflight.log','datacheck_console.log'):
                path=case/name
                if path.exists():z.write(path,path.name)
        temp.replace(archive)
    with zipfile.ZipFile(archive) as z:
        if z.testzip() or 'preflight.sta' not in z.namelist():raise RuntimeError('Invalid preflight archive')
        if b'THE ANALYSIS HAS COMPLETED SUCCESSFULLY' not in z.read('preflight.sta'):
            raise RuntimeError('Preflight archive lacks completion evidence')
    deleted=[]
    for path in case.iterdir():
        if path.is_file() and path.stem=='preflight' and path.suffix in SCRATCH_EXTENSIONS:
            inside(path,case).unlink();deleted.append(path.name)
    if deleted:save(case/'preflight_cleanup.json',dict(deleted=deleted,archive_sha256=digest(archive),utc=now()))


def preflight_increments(sta):
    """Separate the raw element estimate from increment zero after mass scaling."""
    initial=re.search(r'Initial time increment\s*=\s*([0-9.]+E[+-][0-9]+)',sta,re.I)
    rows=re.findall(r'^\s*0\s+[0-9.E+-]+\s+[0-9.E+-]+\s+\d+:\d+:\d+\s+([0-9.]+E[+-][0-9]+)\s+\d+\s+([0-9.]+E[+-][0-9]+)\s+([0-9.E+-]+)(?:\s+([0-9.]+E[+-][0-9]+))?\s*$',sta,re.M|re.I)
    raw=float(initial.group(1)) if initial else None
    scaled=float(rows[-1][0]) if rows else None
    mass=float(rows[-1][3]) if rows and rows[-1][3] else None
    return dict(unscaled_initial_increment_s=raw,scaled_initial_increment_s=scaled,
                initial_mass_change_percent=mass,stable_increment_s=scaled or raw)


def datacheck(case,cfg,budget_minutes=None):
    if (case/'datacheck.json').exists():
        data=read(case/'datacheck.json')
        if not data['passed']:raise RuntimeError('Existing datacheck failed')
        if data['input_sha256']!=digest(case/'analysis.inp'):raise RuntimeError('Datacheck input changed')
        clean_preflight(case)
        return data
    if (case/'preflight.lck').exists():raise RuntimeError('Datacheck is active')
    with lock(ROOT/'runs/.external.lock'):
        # A datacheck needs no parallel solution. Serial execution avoids the
        # MPI initialization stall observed before increment zero in s0007.
        state(case,'DATACHECK_RUNNING',cpus=1,domains=1)
        save(case/'datacheck_attempt.json',dict(input_sha256=digest(case/'analysis.inp'),
             started_utc=now(),cpus=1,domains=1))
        launch([os.environ.get("LOGIC_ABAQUS", cfg['runtime']['abaqus']),'job=preflight','input=analysis.inp','datacheck','cpus=1','domains=1',
                'double=both','interactive'],case,'datacheck_console.log',stage_minutes(cfg,budget_minutes))
    sta=(case/'preflight.sta').read_text(errors='replace') if (case/'preflight.sta').exists() else ''
    dat=(case/'preflight.dat').read_text(errors='replace') if (case/'preflight.dat').exists() else ''
    complete='THE ANALYSIS HAS COMPLETED SUCCESSFULLY' in sta
    candidates=re.findall(r'(?:stable time increment|STABLE TIME INCREMENT)[^\n]*?([0-9]+\.[0-9]+E[+-][0-9]+)',sta,re.I)
    if not candidates:
        candidates=re.findall(r'(?:Initial time increment|TIME INCREMENT)[^\n]*?([0-9]+\.[0-9]+E[+-][0-9]+)',sta,re.I)
    increments=preflight_increments(sta)
    if increments['stable_increment_s'] is None:
        increments['stable_increment_s']=float(candidates[0]) if candidates else None
    data=dict(passed=complete and '***ERROR' not in dat.upper(),warnings=[x.strip() for x in dat.splitlines() if '***WARNING' in x.upper()],
              **increments,input_sha256=digest(case/'analysis.inp'))
    save(case/'datacheck.json',data);state(case,'DATACHECK_OK' if data['passed'] else 'DATACHECK_FAILED')
    if not data['passed']:raise RuntimeError('Abaqus datacheck failed')
    clean_preflight(case)
    if data['stable_increment_s']:
        analysis=read(case/'analysis.json'); dt=data['stable_increment_s']
        save(case/'cost_estimate.json',dict(initial_stable_increment_s=dt,
             unscaled_initial_increment_s=data['unscaled_initial_increment_s'],
             scaled_initial_increment_s=data['scaled_initial_increment_s'],
             initial_mass_change_percent=data['initial_mass_change_percent'],
             initial_increment_estimate=analysis['step_time_s']/dt,
             step_time_s=analysis['step_time_s'],mean_speed_mm_s=analysis['mean_speed_mm_s'],
             scope='Initial increment count only; wall time needs a measured pilot. Timestep changes during deformation.'))
    return data


def solver_command(cfg):
    threads=cfg['solver'].get('threads_per_mpi_process',1)
    cpus=cfg['solver']['cpus'];domains=cfg['solver']['domains']
    if isinstance(threads,bool) or not isinstance(threads,int) or threads<1 or cpus%threads:
        raise ValueError('Threads per MPI process must divide the CPU count')
    command=[os.environ.get("LOGIC_ABAQUS", cfg['runtime']['abaqus']),'job=analysis','input=analysis.inp','cpus='+str(cpus),'domains='+str(domains),
             'double=both','output_precision=full','memory=80%','interactive']
    if threads>1:command.append('threads_per_mpi_process='+str(threads))
    return command


def effective_solver_config(case,cfg):
    path=case/'solver_execution.json'
    if not path.exists():return cfg
    execution=read(path)
    if execution['input_sha256']!=read(case/'analysis.json')['input_sha256']:
        raise RuntimeError('Solver retry belongs to a different input')
    if (execution['cpus'],execution['domains'],execution['threads_per_mpi_process'])!=(8,8,1):
        raise RuntimeError('Unsupported solver retry configuration')
    effective=copy.deepcopy(cfg)
    for name in ('cpus','domains','threads_per_mpi_process'):effective['solver'][name]=execution[name]
    return effective


def solve(case,cfg,budget_minutes):
    receipt=read(case/'datacheck.json')
    if not receipt['passed'] or digest(case/'analysis.inp')!=receipt['input_sha256']:
        raise RuntimeError('Datacheck or frozen input mismatch')
    if (case/'analysis.sta').exists():
        text=(case/'analysis.sta').read_text(errors='replace')
        if 'THE ANALYSIS HAS COMPLETED SUCCESSFULLY' in text and not (case/'analysis.lck').exists():return
        raise RuntimeError('Existing solver output preserved; inspect status and resume exact job if suspended')
    if shutil.disk_usage(case).free<cfg['runtime']['minimum_free_gb']*1024**3:
        raise RuntimeError('Insufficient scratch disk space')
    with lock(ROOT/'runs/.external.lock'):
        effective=effective_solver_config(case,cfg)
        state(case,'SOLVING',cpus=effective['solver']['cpus'],domains=effective['solver']['domains'])
        launch(solver_command(effective),case,'solve_console.log',budget_minutes,solver_job='analysis')
    sta=(case/'analysis.sta').read_text(errors='replace')
    if 'THE ANALYSIS HAS COMPLETED SUCCESSFULLY' not in sta:
        state(case,'SOLVER_FAILED');raise RuntimeError('Solver did not complete; extract diagnostic history before cleanup')
    state(case,'SOLVED')


def resume(case,cfg,budget_minutes):
    case=inside(case)
    with lock(ROOT/'runs/.external.lock'):
        active=active_external()
        if not active or active['case']!=str(case) or active.get('solver_job')!='analysis':
            raise RuntimeError('No tracked active analysis job to resume in this exact case')
        if read(case/'state.json')['stage'] not in ('BUDGET_SUSPEND_REQUESTED','USER_SUSPEND_REQUESTED'):
            raise RuntimeError('Resume requires a recorded suspension request; inspect status')
        verify_files(case,{'analysis.inp':read(case/'analysis.json')['input_sha256']})
        start=time.monotonic()
        with (case/'resume_console.log').open('a',encoding='utf-8') as log:
            result=subprocess.run([os.environ.get("LOGIC_ABAQUS", cfg['runtime']['abaqus']),'resume','job=analysis'],cwd=case,
                    stdout=log,stderr=subprocess.STDOUT,timeout=45,**hidden_options())
            if result.returncode:raise RuntimeError('Resume command failed')
            state(case,'SOLVING_RESUMED')
            try:
                while active_external():
                    if time.monotonic()-start>budget_minutes*60:
                        subprocess.run([os.environ.get("LOGIC_ABAQUS", cfg['runtime']['abaqus']),'suspend','job=analysis'],cwd=case,
                              stdout=log,stderr=subprocess.STDOUT,timeout=45,**hidden_options())
                        state(case,'BUDGET_SUSPEND_REQUESTED',job='analysis')
                        raise TimeoutError('Resume budget reached; suspend requested for this job')
                    time.sleep(1)
            except KeyboardInterrupt:
                try:
                    subprocess.run([os.environ.get("LOGIC_ABAQUS", cfg['runtime']['abaqus']),'suspend','job=analysis'],cwd=case,
                          stdout=log,stderr=subprocess.STDOUT,timeout=45,**hidden_options())
                finally:state(case,'USER_SUSPEND_REQUESTED',job='analysis')
                raise
        sta=(case/'analysis.sta').read_text(errors='replace')
        if 'THE ANALYSIS HAS COMPLETED SUCCESSFULLY' not in sta:
            state(case,'SOLVER_FAILED');raise RuntimeError('Resumed solver did not complete')
        state(case,'SOLVED')


def extract(case,cfg,budget_minutes=None):
    if (case/'extraction.json').exists():return read(case/'extraction.json')
    with lock(ROOT/'runs/.external.lock'):
        launch([os.environ.get("LOGIC_ABAQUS", cfg['runtime']['abaqus']),'python',WORKER_SOURCE/'extract_odb.py',case],case,'extract_console.log',stage_minutes(cfg,budget_minutes))
    if not (case/'extraction.json').exists():
        raise RuntimeError('Abaqus Python did not produce an extraction receipt; inspect extract_console.log')
    state(case,'EXTRACTED');return read(case/'extraction.json')


def cleanup(case):
    case=inside(case)
    if (case/'analysis.lck').exists():raise RuntimeError('Cannot clean an active ODB')
    manifest_path=case/'export_manifest.json'
    diagnostic=not manifest_path.exists()
    if diagnostic:manifest_path=case/'failed_export_manifest.json'
    manifest=read(manifest_path);verify_files(case,manifest['hashes'])
    # Semantic checks are repeated before removing the only database.
    import numpy as np
    if diagnostic:
        if read(case/'extraction.json')['solver_completed']:raise RuntimeError('Complete jobs require full label validation')
        if read(case/'failure_diagnostics.json')['status']!='REJECTED_INCOMPLETE':raise RuntimeError('Missing failure diagnosis')
    else:
        curve=np.load(case/'curve.npz')
        if curve['stress_MPa'].shape!=(1001,) or not np.isfinite(curve['stress_MPa']).all():
            raise RuntimeError('Exported curve is invalid')
    archive=case/'solver_archive.zip'
    # Preserve a complete archive across a resumed cleanup; never overwrite it after deleting inputs.
    if not archive.exists():
        temp=archive.with_suffix('.zip.tmp')
        with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED) as z:
            for p in case.iterdir():
                if p.is_file() and (p.suffix in ('.inp','.sta','.msg','.dat','.log') or '.msg.' in p.name):
                    z.write(p,p.name)
        temp.replace(archive)
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None or 'analysis.inp' not in z.namelist():raise RuntimeError('Invalid solver archive')
        import hashlib
        if hashlib.sha256(z.read('analysis.inp')).hexdigest()!=read(case/'analysis.json')['input_sha256']:
            raise RuntimeError('Archived input differs')
    deleted=[]
    extensions=SCRATCH_EXTENSIONS
    for p in case.iterdir():
        if p.is_file() and p.stem in ('analysis','preflight') and p.suffix in extensions:
            inside(p,case).unlink();deleted.append(p.name)
    old_deleted=read(case/'cleanup_receipt.json').get('deleted',[]) if (case/'cleanup_receipt.json').exists() else []
    save(case/'cleanup_receipt.json',dict(utc=now(),deleted=sorted(set(old_deleted+deleted)),archive_sha256=digest(archive),
                                         export_manifest_sha256=digest(manifest_path),diagnostic_only=diagnostic))
    state(case,'CLEANED_REJECTED' if diagnostic else 'CLEANED',numerical_pass=False if diagnostic else read(case/'quality.json')['passed'])


def archive_failure(case):
    import csv
    import json
    case=inside(case)
    if (case/'analysis.lck').exists():raise RuntimeError('Failed job is still active')
    extraction=read(case/'extraction.json')
    if extraction['solver_completed']:raise ValueError('Use full postprocessing for completed jobs')
    with gzip.open(case/'raw_history.csv.gz','rt',encoding='utf-8') as f:rows=list(csv.DictReader(f))
    if len(rows)<2 or len(rows)!=extraction['samples']:raise RuntimeError('Incomplete diagnostic history export')
    for name in ('energy_history.json.gz','sparse_displacements.json.gz'):
        with gzip.open(case/name,'rt',encoding='utf-8') as f:json.load(f)
    read(case/'field_diagnostics.json')
    config=read(case/'analysis.json')['config']
    last=rows[-1]
    save(case/'failure_diagnostics.json',dict(status='REJECTED_INCOMPLETE',training_eligible=False,
         history_samples=len(rows),last_time_s=float(last['time_s']),
         final_strain=(float(last['bottom_u3_mm'])-float(last['top_u3_mm']))/config['geometry']['size_mm'],
         reason='Solver did not complete; partial curves are diagnostic data only'))
    names=['raw_history.csv.gz','energy_history.json.gz','field_diagnostics.json','sparse_displacements.json.gz',
           'extraction.json','analysis.json','failure_diagnostics.json']
    save(case/'failed_export_manifest.json',dict(hashes={name:digest(case/name) for name in names}))
    cleanup(case)
