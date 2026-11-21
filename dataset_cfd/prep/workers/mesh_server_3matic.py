"""Resident 3-matic request loop. Each result follows an empty-project check."""
import ctypes
import gc
import json
import os
from pathlib import Path
import sys
import time
import traceback
import trimatic

sys.path.insert(0,str(Path(__file__).resolve().parent))
from mesh_3matic import process_case


def save(path,value):
    temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value,indent=2),encoding='utf-8');temp.replace(path)


def owner_alive(owner):
    # The embedded interpreter need not have psutil. Check PID creation time on Windows.
    if os.name!='nt':
        try:os.kill(owner['pid'],0);return True
        except ProcessLookupError:return False
    from ctypes import wintypes
    api=ctypes.WinDLL('kernel32',use_last_error=True)
    api.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD]
    api.OpenProcess.restype=wintypes.HANDLE
    api.GetProcessTimes.argtypes=[wintypes.HANDLE]+[ctypes.POINTER(wintypes.FILETIME)]*4
    api.WaitForSingleObject.argtypes=[wintypes.HANDLE,wintypes.DWORD]
    api.CloseHandle.argtypes=[wintypes.HANDLE]
    handle=api.OpenProcess(0x100000|0x1000,False,owner['pid'])
    if not handle:return False
    try:
        times=[wintypes.FILETIME() for _ in range(4)]
        if not api.GetProcessTimes(handle,*[ctypes.byref(t) for t in times]):return False
        created=((times[0].dwHighDateTime<<32)|times[0].dwLowDateTime)/1e7-11644473600
        return abs(created-owner['created'])<.01 and api.WaitForSingleObject(handle,0)==258
    finally:api.CloseHandle(handle)


def serve(folder):
    owner=json.loads((folder/'owner.json').read_text(encoding='utf-8'))
    run=folder.parent.parent;request_path=folder/'request.json';last=None
    trimatic.new_project()
    if trimatic.get_parts():raise RuntimeError('Initial project is not empty')
    save(folder/'ready.json',dict(status='READY',pid=os.getpid()))
    while owner_alive(owner) and not (folder/'stop.json').exists():
        if not request_path.exists():time.sleep(.2);continue
        request=json.loads(request_path.read_text(encoding='utf-8'))
        if request['id']==last:time.sleep(.2);continue
        last=request['id'];case=Path(request['case']).resolve()
        if not case.is_relative_to((run/'samples').resolve()):raise RuntimeError('Case outside session run')
        session=dict(directory=str(folder),pid=os.getpid(),request_id=last,sequence=request['sequence'])
        try:
            result=process_case(case,session)
        except Exception:
            result=dict(status='FAILED',error=traceback.format_exc(),session=session,cleanup_failed=True)
            save(case/'mesher_result.json',result)
        gc.collect()
        retire=result['status']!='MESH_EXPORTED'
        save(folder/'response.json',dict(id=last,case=str(case),status=result['status'],retire=retire))
        if retire:break


if __name__=='__main__':
    try:serve(Path(os.environ['FEA_V2_SESSION']).resolve())
    except Exception:traceback.print_exc()
    finally:trimatic.exit()
