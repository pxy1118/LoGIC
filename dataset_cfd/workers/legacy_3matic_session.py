"""Resident 3-matic host; execute the unchanged native script once per request."""
import gc
import hashlib
import json
import os
from pathlib import Path
import runpy
import time
import traceback


def read_json(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    path=Path(path);temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8');temp.replace(path)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def process_request(request, session, root, native, runner=runpy.run_path):
    """Per-case exceptions are returned; they do not exit the native host."""
    case=Path(request['case']).resolve();worker=root/'workers/legacy_remesh_3matic.py'
    reply={'id':request['id'],'case':str(case),'pid':os.getpid(),'status':'FAILED'}
    previous=os.environ.get('CFD_V2_CASE');original_exit=native.exit
    try:
        if not case.is_relative_to(root/'runs'):raise ValueError('Case outside CFD runs')
        cfg=read_json(session/'manager.json')
        if sha(case/'protocol.json')!=request['protocol_sha256'] or cfg['protocol_sha256']!=request['protocol_sha256']:
            raise RuntimeError('Session/case protocol changed')
        if sha(worker)!=request['remesh_sha256'] or cfg['remesh_sha256']!=request['remesh_sha256']:
            raise RuntimeError('Native remesh script changed')
        os.environ['CFD_V2_CASE']=str(case)
        # The original script calls new_project() before importing a case.
        # Suppress only its final application exit, keeping all remesh calls.
        native.exit=lambda:None
        runner(str(worker),run_name='__main__')
        reply.update(status='COMPLETE',report_sha256=sha(case/'remesh_report.json'))
    except BaseException:
        reply['error']=traceback.format_exc();traceback.print_exc()
    finally:
        native.exit=original_exit
        if previous is None:os.environ.pop('CFD_V2_CASE',None)
        else:os.environ['CFD_V2_CASE']=previous
        gc.collect()
    return reply


def main():
    import trimatic
    root=Path(__file__).resolve().parents[1]
    session=Path(os.environ['CFD_V2_3MATIC_SESSION']).resolve()
    if not session.is_relative_to(root/'runs'):raise ValueError('Session outside CFD runs')
    write_json(session/'ready.json',{'status':'READY','pid':os.getpid()})
    idle_since=time.monotonic()
    try:
        while not (session/'shutdown.json').exists():
            requests=[p for p in sorted((session/'requests').glob('*.json'))
                      if not (session/'responses'/p.name).exists()]
            if not requests:
                write_json(session/'worker_state.json',{'phase':'IDLE','pid':os.getpid()})
                if time.monotonic()-idle_since>600:break
                time.sleep(.25);continue
            for path in requests:
                request=read_json(path)
                print('SESSION_CASE_START',request['case'],request['id'],flush=True)
                write_json(session/'worker_state.json',{'phase':'WORKING','pid':os.getpid(),
                           'case':request['case'],'request_id':request['id']})
                reply=process_request(request,session,root,trimatic)
                write_json(session/'responses'/path.name,reply)
                print('SESSION_CASE_END',request['case'],reply['status'],flush=True)
                idle_since=time.monotonic()
    finally:
        write_json(session/'worker_state.json',{'phase':'CLOSED','pid':os.getpid()})
        trimatic.exit()


if __name__=='__main__':main()
