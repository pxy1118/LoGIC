"""Run one isolated legacy-aligned geometry/CFD smoke case with receipts."""
from pathlib import Path
import argparse
import os
import sys
import time
import shutil
import subprocess
import traceback
import psutil
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.common import ROOT,read,save,digest,lock,state,now
from src.geometry import prepare_solid,report
from src.surface_io import read_ascii_stl64

parser=argparse.ArgumentParser()
parser.add_argument('--sample',type=int,default=68)
parser.add_argument('--run',default='legacy_aligned_pilot_20261107')
parser.add_argument('--geometry-only',action='store_true')
parser.add_argument('--profile',type=Path,default=ROOT/'configs/legacy_aligned_candidate.json')
args=parser.parse_args()
if not args.run.replace('_','').replace('-','').isalnum():raise ValueError('Invalid run')
run=ROOT/'runs'/args.run
case=run/'samples'/str(args.sample)
if case.exists():raise RuntimeError('Existing case preserved; choose a new run for another attempt')
case.mkdir(parents=True)
cfg_path=args.profile.resolve()
shutil.copy2(cfg_path,case/'protocol.json')
cfg=read(case/'protocol.json')
snapshot=case/'executed_source';snapshot.mkdir()
sources=[*sorted((ROOT/'src').glob('*.py')),Path(__file__),
         ROOT/'workers/legacy_fluid_blender.py',ROOT/'workers/legacy_remesh_3matic.py']
hashes={}
for path in sources:
    dest=snapshot/path.name;shutil.copy2(path,dest);hashes[str(path.relative_to(ROOT))]=digest(dest)
save(snapshot/'manifest.json',hashes)
receipt={'status':'RUNNING','pid':os.getpid(),'created':psutil.Process().create_time(),
         'utc':now(),'sample_id':args.sample,'run':args.run,
         'protocol_sha256':digest(case/'protocol.json'),'stages':[]}
save(case/'process.json',receipt)
started=time.time()


def native(command,name,timeout_s=900):
    if name=='remesh' and os.environ.get('CFD_V2_3MATIC_SESSION'):
        from src.trimatic_session import submit_case
        binding=submit_case(case,timeout_s)
        receipt['stages'].append({'name':name,'command':command,'exit_code':0,
             'elapsed_s':binding['elapsed_s'],'execution_mode':'persistent_3matic',
             'session_dir':binding['session_dir'],'session_pid':binding['pid'],'request_id':binding['id']})
        save(case/'process.json',receipt)
        return
    env=os.environ.copy();env['CFD_V2_CASE']=str(case)
    item={'name':name,'command':command,'started_utc':now()};t=time.time()
    startup=subprocess.STARTUPINFO();startup.dwFlags|=subprocess.STARTF_USESHOWWINDOW
    startup.wShowWindow=0
    with (case/(name+'.stdout.log')).open('w') as out,(case/(name+'.stderr.log')).open('w') as err:
        child=subprocess.Popen(command,cwd=ROOT.parent,env=env,stdout=out,stderr=err,
                               creationflags=subprocess.CREATE_NO_WINDOW,startupinfo=startup)
        proc=psutil.Process(child.pid)
        save(case/'native_process.json',{'status':'RUNNING','pid':child.pid,
             'created':proc.create_time(),'name':name,'command':command})
        try:code=child.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            state(case,'NATIVE_TIMEOUT_PROCESS_RETAINED',pid=child.pid,name=name)
            raise RuntimeError('Native stage timed out; process retained for inspection')
    save(case/'native_process.json',{'status':'FINISHED','pid':child.pid,'name':name,'exit_code':code})
    item.update(exit_code=code,elapsed_s=time.time()-t)
    receipt['stages'].append(item);save(case/'process.json',receipt)
    if code:raise RuntimeError('Native stage failed: '+name)


try:
    with lock(run/'.writer.lock'),lock(case/'.case.lock'):
        import os as _os
        _os.environ['CFD_V2_SOLID_SOURCE']=str(cfg['geometry'].get('solid_source','inp_surface'))
        prepare_solid(case,args.sample)
        working_scale=cfg['geometry'].get('processing_scale',1.0)
        if working_scale!=1:
            import pyvista as pv
            working=pv.read(case/'solid_initial.stl');working.points*=working_scale
            working.save(case/'solid_processing.stl')
        if cfg['geometry']['boolean_backend']=='Blender EXACT':
            state(case,'BLENDER_BOOLEAN')
            native([os.environ.get("LOGIC_BLENDER", "blender"),'--background','--python-exit-code','1',
                    '--python',str(ROOT/'workers/legacy_fluid_blender.py')],'blender')
            if read(case/'blender_report.json')['status']!='BOOLEAN_EXPORTED':
                raise RuntimeError('No successful Blender result')
        state(case,'SURFACE_REMESHING')
        if os.environ.get('CFD_V2_3MATIC_SESSION'):
            native([os.environ.get("LOGIC_TRIMATIC", "3-matic.exe"),
                    '-run_script',str(ROOT/'workers/legacy_remesh_3matic.py')],'remesh')
        else:
            with lock(ROOT/'.geometry_native.lock'):
                native([os.environ.get("LOGIC_TRIMATIC", "3-matic.exe"),
                    '-run_script',str(ROOT/'workers/legacy_remesh_3matic.py')],'remesh')
        remesh=read(case/'remesh_report.json')
        if remesh['status'] not in ['SURFACE_REMESH_EXPORTED','SURFACE_EXPORTED_WITHOUT_REMESH']:
            raise RuntimeError(remesh.get('error','Remeshing failed'))
        import pyvista as pv
        import numpy as np
        float64_surface=cfg['geometry'].get('surface_export_format')=='ascii_float64'
        if working_scale!=1 and not float64_surface:
            for src,dst in [('fluid_boolean_working.stl','fluid_boolean_raw.stl'),('fluid_working.stl','fluid_domain.stl')]:
                working=pv.read(case/src);working.points/=working_scale;working.save(case/dst)
        reader=read_ascii_stl64 if float64_surface else pv.read
        raw=reader(case/'fluid_boolean_raw.stl').triangulate().clean()
        final=reader(case/'fluid_domain.stl').triangulate().clean()
        r=report(final);raw_report=report(raw)
        d1=np.abs(final.compute_implicit_distance(raw).point_data['implicit_distance'])
        d2=np.abs(raw.compute_implicit_distance(final).point_data['implicit_distance'])
        distances=np.r_[d1,d2]
        closure=abs(r['volume_mm3']-raw_report['volume_mm3'])/cfg['geometry']['specimen_size_mm']**3
        deviation={'method':'symmetric vertex-to-triangle distance','p95':float(np.quantile(distances,.95)),
                   'max':float(distances.max())}
        r.update(raw_surface=raw_report,remesh_volume_change_rel_specimen=closure,
                 symmetric_surface_distance_mm=deviation,sha256=digest(case/'fluid_domain.stl'),
                 raw_sha256=digest(case/'fluid_boolean_raw.stl'),production_geometry_valid=False,
                 surface_export_format=cfg['geometry'].get('surface_export_format','binary_float32'),
                 audited_coordinate_dtype=str(final.points.dtype),
                 config_sha256=digest(case/'protocol.json'))
        r['accepted_for_development']=bool(r['boundary_edges']==0 and r['nonmanifold_edges']==0 and
            closure<=.001 and deviation['p95']<=.005 and deviation['max']<=.02)
        save(case/'fluid_report.json',r)
        if not r['accepted_for_development']:
            raise RuntimeError('Closed-surface or geometry-deviation gate failed; inspect fluid_report.json')
        state(case,'FLUID_READY',**r)
        if not args.geometry_only:
            with lock(ROOT/'.comsol_worker.lock'),lock(ROOT/'runs/pilot_20261107/.solver.lock'):
                from src.legacy_solver import solve_legacy
                result=solve_legacy(case)
                if not result['cfd_valid']:raise RuntimeError('CFD result QA failed')
        receipt.update(status='FINISHED',elapsed_s=time.time()-started,utc=now())
        save(case/'process.json',receipt)
except Exception:
    receipt.update(status='FAILED',elapsed_s=time.time()-started,error=traceback.format_exc(),utc=now())
    save(case/'process.json',receipt);state(case,'LEGACY_PILOT_FAILED',error=receipt['error'])
    raise
