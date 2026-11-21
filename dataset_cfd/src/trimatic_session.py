"""One owned 3-matic process per geometry batch; serialized case requests."""
import os
from pathlib import Path
import subprocess
import time
import uuid
import psutil
from .common import ROOT,read,save,digest,lock,now

SESSION_ENV='CFD_V2_3MATIC_SESSION'
SERVER=ROOT/'workers/legacy_3matic_session.py'
REMESH=ROOT/'workers/legacy_remesh_3matic.py'
EXECUTABLE=os.environ.get("LOGIC_TRIMATIC", "3-matic.exe")


def owned_native(receipt):
    try:
        p=psutil.Process(receipt['pid'])
        if abs(p.create_time()-receipt['created'])>.1:return None
        if str(SERVER).casefold() not in [arg.casefold() for arg in p.cmdline()]:
            raise RuntimeError('Session PID belongs to another command; no process was touched')
        return p
    except psutil.NoSuchProcess:return None


def submit_case(case, timeout_s=900):
    case=Path(case).resolve();session=Path(os.environ[SESSION_ENV]).resolve()
    if not case.is_relative_to(ROOT/'runs') or not session.is_relative_to(ROOT/'runs'):
        raise ValueError('Session and case must stay inside CFD runs')
    owner=read(session/'session.json')
    if not owned_native(owner):raise RuntimeError('Persistent 3-matic session exited')
    request_id=f'{time.time_ns()}_{uuid.uuid4().hex}'
    request={'id':request_id,'case':str(case),'protocol_sha256':digest(case/'protocol.json'),
             'remesh_sha256':digest(REMESH),'case_pid':os.getpid(),
             'case_created':psutil.Process().create_time(),'utc':now()}
    binding={**request,'session_dir':str(session),'pid':owner['pid'],'created':owner['created'],'status':'RUNNING'}
    save(case/'shared_native_request.json',binding)
    save(session/'requests'/f'{request_id}.json',request)
    response=session/'responses'/f'{request_id}.json';started=time.monotonic();strikes=0
    try:
        while not response.exists():
            p=owned_native(owner)
            if not p:raise RuntimeError('Persistent 3-matic crashed during this case')
            low=psutil.virtual_memory().available<8*2**30 or p.memory_info().rss>40*2**30
            strikes=strikes+1 if low else 0
            if strikes>=2:raise RuntimeError('Persistent 3-matic resource limit reached')
            if time.monotonic()-started>timeout_s:raise TimeoutError('Persistent 3-matic request timed out')
            time.sleep(.25)
        reply=read(response)
        if reply['id']!=request_id or reply['case']!=str(case):
            raise RuntimeError('Session response belongs to a different case')
        if reply['status']!='COMPLETE':raise RuntimeError(reply.get('error','Native session request failed'))
        if reply['report_sha256']!=digest(case/'remesh_report.json'):
            raise RuntimeError('Session output changed after native processing')
        binding.update(status='FINISHED',finished_utc=now(),response_sha256=digest(response),
                       elapsed_s=time.monotonic()-started)
        save(case/'shared_native_request.json',binding)
        return binding
    except BaseException:
        binding.update(status='FAILED',finished_utc=now());save(case/'shared_native_request.json',binding)
        raise


class TrimaticSession:
    def __init__(self, root):
        self.root=Path(root).resolve();self.folder=None;self.receipt=None;self.process=None
        self.lease=None;self.streams=[];self.previous_env=os.environ.get(SESSION_ENV)

    def _retire_orphans(self):
        for path in (ROOT/'runs').glob('*/trimatic_sessions/session_*/session.json'):
            record=read(path);native=owned_native(record)
            if native is None:continue
            try:
                manager=psutil.Process(record['batch_pid'])
                if abs(manager.create_time()-record['batch_created'])<.1:
                    raise RuntimeError('Another live batch owns a 3-matic session')
            except psutil.NoSuchProcess:pass
            state_path=path.parent/'worker_state.json'
            if state_path.exists():
                status=read(state_path)
                if status.get('phase')=='WORKING':
                    request=read(path.parent/'requests'/f'{status["request_id"]}.json')
                    try:
                        case_owner=psutil.Process(request['case_pid'])
                        if abs(case_owner.create_time()-request['case_created'])<.1:
                            raise RuntimeError('An orphaned session still has a live case worker; let it finish')
                    except psutil.NoSuchProcess:pass
            native.terminate()
            try:native.wait(timeout=15)
            except psutil.TimeoutExpired:
                native=owned_native(record)
                if native:native.kill();native.wait(timeout=15)
            record.update(status='ORPHAN_RETIRED',finished_utc=now())
            save(path,record)

    def ensure_started(self):
        if self.receipt and owned_native(self.receipt):return
        if self.receipt and self.receipt['status'] not in ('STOPPED_RETAINED','CLOSED'):
            self.receipt.update(status='EXITED_RETAINED',finished_utc=now())
            save(self.folder/'session.json',self.receipt)
        if self.lease is None:
            self.lease=lock(ROOT/'.geometry_native.lock');self.lease.__enter__()
            self._retire_orphans()
        folders=self.root/'trimatic_sessions';folders.mkdir(exist_ok=True)
        n=1
        while (folders/f'session_{n:04d}').exists():n+=1
        self.folder=folders/f'session_{n:04d}';self.folder.mkdir()
        (self.folder/'requests').mkdir();(self.folder/'responses').mkdir()
        env=os.environ.copy();env[SESSION_ENV]=str(self.folder)
        owner={'batch_pid':os.getpid(),'batch_created':psutil.Process().create_time(),
               'run_root':str(self.root),'server_sha256':digest(SERVER),
               'remesh_sha256':digest(REMESH),'protocol_sha256':digest(self.root/'protocol.json')}
        save(self.folder/'manager.json',owner)
        out=(self.folder/'native.stdout.log').open('wb');err=(self.folder/'native.stderr.log').open('wb')
        self.streams.extend([out,err])
        startup=subprocess.STARTUPINFO();startup.dwFlags|=subprocess.STARTF_USESHOWWINDOW;startup.wShowWindow=0
        self.process=subprocess.Popen([EXECUTABLE,'-run_script',str(SERVER)],cwd=ROOT.parent,env=env,
            stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW,startupinfo=startup)
        p=psutil.Process(self.process.pid)
        self.receipt={**owner,'status':'STARTING','pid':p.pid,'created':p.create_time(),'started_utc':now()}
        save(self.folder/'session.json',self.receipt)
        started=time.monotonic()
        while not (self.folder/'ready.json').exists():
            if not owned_native(self.receipt):raise RuntimeError('3-matic session failed to start')
            if time.monotonic()-started>90:raise TimeoutError('3-matic session startup timed out')
            time.sleep(.25)
        if read(self.folder/'ready.json')['pid']!=p.pid:raise RuntimeError('Session readiness PID mismatch')
        self.receipt['status']='RUNNING';save(self.folder/'session.json',self.receipt)
        os.environ[SESSION_ENV]=str(self.folder)

    def _stop(self, reason):
        if not self.receipt:return
        p=owned_native(self.receipt)
        if p:
            p.terminate()
            try:p.wait(timeout=15)
            except psutil.TimeoutExpired:
                p=owned_native(self.receipt)
                if p:p.kill();p.wait(timeout=15)
        self.receipt.update(status='STOPPED_RETAINED',reason=reason,finished_utc=now())
        save(self.folder/'session.json',self.receipt)

    def recover_after_case(self, case):
        if not self.folder:return
        path=Path(case)/'shared_native_request.json'
        if not path.exists():return
        binding=read(path)
        if binding['session_dir']!=str(self.folder):raise RuntimeError('Case used another native session')
        response=self.folder/'responses'/f'{binding["id"]}.json'
        if response.exists():return
        # The case worker exited/was interrupted while the native call is
        # still in flight. Restart only this verified session, not other apps.
        self._stop('ABANDONED_OR_TIMED_OUT_CASE')
        binding.update(status='ABORTED_RETAINED',finished_utc=now());save(path,binding)

    def close(self):
        try:
            if self.receipt and owned_native(self.receipt):
                save(self.folder/'shutdown.json',{'utc':now(),'reason':'BATCH_COMPLETE_OR_STOPPED'})
                try:self.process.wait(timeout=15)
                except subprocess.TimeoutExpired:self._stop('SESSION_SHUTDOWN_TIMEOUT')
                if self.receipt['status']!='STOPPED_RETAINED':
                    self.receipt.update(status='CLOSED',finished_utc=now());save(self.folder/'session.json',self.receipt)
            elif self.receipt and self.receipt['status'] not in ('CLOSED','STOPPED_RETAINED'):
                self.receipt.update(status='EXITED_RETAINED',finished_utc=now());save(self.folder/'session.json',self.receipt)
        finally:
            if self.previous_env is None:os.environ.pop(SESSION_ENV,None)
            else:os.environ[SESSION_ENV]=self.previous_env
            for stream in self.streams:stream.close()
            if self.lease is not None:self.lease.__exit__(None,None,None);self.lease=None
