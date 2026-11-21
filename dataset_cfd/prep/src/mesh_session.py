"""One resident 3-matic process, with sequential requests and durable ownership."""
import os
from pathlib import Path
import subprocess
import time
import uuid
import psutil
from .common import ROOT,inside,read,save,now,lock


class MeshSession:
    MAX_CASES=50
    MAX_RSS_BYTES=8*1024**3

    def __init__(self,run,cfg):
        self.run=inside(run);self.cfg=cfg;self.process=None;self.folder=None
        self.count=0;self.busy=False;self.log=None;self.identities={}
        owner=psutil.Process()
        self.owner=dict(pid=owner.pid,created=owner.create_time())

    def register(self,case=None):
        from . import runtime
        try:
            parent=psutil.Process(self.process.pid)
            for process in [parent]+parent.children(recursive=True):
                try:self.identities[process.pid]=dict(pid=process.pid,created=process.create_time())
                except psutil.NoSuchProcess:pass
        except psutil.NoSuchProcess:pass
        save(runtime.REGISTRY,dict(status='RUNNING',kind='meshing_session',owner=self.owner,
             busy=self.busy,case=str(case or self.folder),session_dir=str(self.folder),
             command=self.command,solver_job=None,processes=list(self.identities.values()),updated_utc=now()))

    def alive(self):
        """The launcher may exit before its native child; track both identities."""
        self.process.poll()
        for identity in self.identities.values():
            try:
                p=psutil.Process(identity['pid'])
                if abs(p.create_time()-identity['created'])<.01 and p.is_running():return True
            except psutil.NoSuchProcess:pass
        return False

    def wait_file(self,name,deadline):
        path=self.folder/name;next_register=0
        while not path.exists():
            if time.monotonic()>=next_register:
                self.register(self.current_case if self.busy else None);next_register=time.monotonic()+2
            if not self.alive():raise RuntimeError('Resident 3-matic exited; inspect '+str(self.folder/'console.log'))
            if time.monotonic()>=deadline:raise TimeoutError('Resident 3-matic budget reached; active job preserved')
            time.sleep(.2)
        return read(path)

    def start(self,deadline):
        from . import runtime
        runtime.assert_external_idle()
        self.folder=self.run/'meshing_sessions'/uuid.uuid4().hex
        self.folder.mkdir(parents=True)
        save(self.folder/'owner.json',self.owner)
        self.command=[os.environ.get("LOGIC_TRIMATIC", self.cfg['runtime']['trimatic']),'-run_script',str(runtime.WORKER_SOURCE/'mesh_server_3matic.py')]
        env=os.environ.copy();env['FEA_V2_SESSION']=str(self.folder)
        self.log=(self.folder/'console.log').open('w',encoding='utf-8')
        self.process=subprocess.Popen(self.command,cwd=self.folder,stdout=self.log,stderr=subprocess.STDOUT,
                                      env=env,**runtime.hidden_options())
        self.count=0;self.identities={};self.busy=False
        self.register()
        try:
            ready=self.wait_file('ready.json',deadline)
            if ready.get('status')!='READY':raise RuntimeError('3-matic session initialization failed')
        except BaseException:
            # No sample is running: request a clean shutdown, without killing a process tree.
            self.close(locked=True)
            raise

    def rss(self):
        total=0
        for identity in self.identities.values():
            try:
                p=psutil.Process(identity['pid'])
                if abs(p.create_time()-identity['created'])<.01:total+=p.memory_info().rss
            except psutil.NoSuchProcess:pass
        return total

    def mesh(self,case,minutes):
        from . import runtime
        case=inside(case,self.run/'samples')
        if minutes<=0:raise TimeoutError('Budget exhausted before meshing')
        started=time.monotonic();deadline=started+minutes*60
        if self.process is not None and (not self.alive() or self.count>=self.MAX_CASES or self.rss()>self.MAX_RSS_BYTES):
            self.close(locked=True)
        if self.process is None:self.start(deadline)
        request_id=uuid.uuid4().hex
        response=self.folder/'response.json'
        if response.exists():response.unlink()
        self.current_case=case;self.busy=True;self.register(case)
        save(case/'process.json',dict(pid=self.process.pid,command=self.command,started_utc=now(),
             session_dir=str(self.folder),request_id=request_id))
        save(self.folder/'request.json',dict(id=request_id,case=str(case),sequence=self.count+1))
        try:
            answer=self.wait_file('response.json',deadline)
            if answer.get('id')!=request_id or answer.get('case')!=str(case):
                raise RuntimeError('Resident mesher response identity mismatch')
            if not (case/'mesher_result.json').exists():raise RuntimeError('Resident mesher did not write its result')
            result=read(case/'mesher_result.json')
            if result.get('session',{}).get('request_id')!=request_id:raise RuntimeError('Mesher result identity mismatch')
            self.busy=False;self.count+=1;self.register()
            save(case/'mesher_console.log.receipt.json',dict(command=self.command,returncode=0,
                 elapsed_s=time.monotonic()-started,session_dir=str(self.folder),request_id=request_id,
                 session_sequence=self.count,rss_bytes=self.rss()))
            if answer.get('retire'):self.close(locked=True)
        except BaseException:
            # Never submit another request or prune inputs while this job is unresolved.
            if self.busy:
                save(self.folder/'stop.json',dict(reason='request interrupted',utc=now()))
                self.register(case);state='EXTERNAL_PROCESS_RUNNING' if self.alive() else 'MESH_INTERRUPTED'
                runtime.state(case,state,pid=self.process.pid,session_dir=str(self.folder))
                if state=='MESH_INTERRUPTED':
                    self.busy=False;self.close(locked=True)
            raise

    def close(self,locked=False):
        if self.process is None:return
        if not locked:
            with lock(ROOT/'runs/.external.lock'):return self.close(locked=True)
        from . import runtime
        save(self.folder/'stop.json',dict(reason='coordinator shutdown or session rotation',utc=now()))
        if self.busy:
            # Worker finishes this sample, then sees stop.json and exits on its own.
            self.register(self.current_case)
            self.log.close();self.log=None
            return
        deadline=time.monotonic()+15
        while self.alive() and time.monotonic()<deadline:time.sleep(.2)
        if self.alive():
            self.register()
            raise RuntimeError('Resident mesher has not exited; inspect session before restarting')
        save(self.folder/'closed.json',dict(utc=now(),completed_cases=self.count))
        save(runtime.REGISTRY,dict(read(runtime.REGISTRY),status='ENDED',checked_utc=now()))
        if self.log is not None:self.log.close()
        self.log=None;self.process=None;self.identities={}
