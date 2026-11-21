"""Start a new solver attempt from an accepted, hash-verified geometry cache."""
from pathlib import Path
import argparse
import os
import sys
import shutil
import time
import traceback
import psutil
sys.stdout.reconfigure(encoding='utf8');sys.stderr.reconfigure(encoding='utf8')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.common import ROOT,read,save,digest,state,lock,now
from src.legacy_protocol import validate_cached_mesh_variant

p=argparse.ArgumentParser()
p.add_argument('--source-case',type=Path,required=True)
p.add_argument('--run',required=True)
p.add_argument('--reuse-mesh',action='store_true')
p.add_argument('--profile',type=Path,help='Volume-mesh variant; geometry and physical conditions must match')
a=p.parse_args()
source=a.source_case.resolve()
if not source.is_relative_to(ROOT/'runs'):raise ValueError('Source outside CFD runs')
if not a.run.replace('_','').replace('-','').isalnum():raise ValueError('Invalid run')
geometry=read(source/'fluid_report.json')
if not geometry['accepted_for_development']:raise RuntimeError('Source geometry was not accepted')
if digest(source/'fluid_domain.stl')!=geometry['sha256']:
    raise RuntimeError('Source geometry hash mismatch')
if digest(source/'protocol.json')!=geometry['config_sha256']:
    raise RuntimeError('Source geometry configuration changed')
original_protocol=read(source/'protocol.json')
protocol=read(a.profile.resolve()) if a.profile else original_protocol
validate_cached_mesh_variant(original_protocol,protocol,reuse_mesh=a.reuse_mesh)
old=read(source/'process.json')
try:
    proc=psutil.Process(old['pid'])
    if abs(proc.create_time()-old['created'])<.1:
        raise RuntimeError('Source owner is still live; do not copy an active case')
except psutil.NoSuchProcess:pass
sample=read(source/'source.json')['sample_id']
case=ROOT/'runs'/a.run/'samples'/str(sample)
case.mkdir(parents=True,exist_ok=False)
for name in ['protocol.json','source.json','solid_initial.stl','solid_report.json',
             'fluid_domain.stl','fluid_report.json','fluid_boolean_raw.stl',
             'blender_report.json','remesh_report.json']:
    if (source/name).exists():shutil.copy2(source/name,case/name)
if a.profile:
    shutil.copy2(source/'protocol.json',case/'geometry_origin_protocol.json')
    # Keep the exact frozen bytes, including line endings: the staged batch
    # verifies file hashes, not only dictionary equality.
    shutil.copy2(a.profile.resolve(),case/'protocol.json')
    if read(case/'protocol.json')!=protocol:
        raise RuntimeError('Override profile changed while it was copied')
    geometry=dict(geometry)
    geometry.update(geometry_origin_protocol_sha256=digest(source/'protocol.json'),
                    geometry_origin_report_sha256=digest(source/'fluid_report.json'),
                    config_sha256=digest(case/'protocol.json'),
                    reuse_reason='Identical geometry, source and physics; volume-mesh variant only')
    save(case/'fluid_report.json',geometry)
snapshot=case/'executed_source';snapshot.mkdir()
code={}
for src in [*sorted((ROOT/'src').glob('*.py')),Path(__file__)]:
    dest=snapshot/src.name;shutil.copy2(src,dest);code[str(src.relative_to(ROOT))]=digest(dest)
save(snapshot/'manifest.json',code)
receipt={'status':'RUNNING','pid':os.getpid(),'created':psutil.Process().create_time(),
         'utc':now(),'sample_id':sample,'protocol_sha256':digest(case/'protocol.json'),
         'geometry_source_case':str(source),'geometry_execution_manifest_sha256':digest(source/'executed_source/manifest.json')}
if a.profile:
    receipt.update(profile_override_path=str(a.profile.resolve()),
                   profile_override_sha256=digest(a.profile.resolve()))
save(case/'process.json',receipt)
started=time.time()
try:
    with lock(case/'.case.lock'),lock(ROOT/'.comsol_worker.lock'),lock(ROOT/'runs/pilot_20261107/.solver.lock'):
        state(case,'REUSED_VERIFIED_GEOMETRY',source_case=str(source),fluid_sha256=geometry['sha256'])
        from src.legacy_solver import solve_legacy,solve_from_native_mesh
        result=solve_from_native_mesh(case,source) if a.reuse_mesh else solve_legacy(case)
        if not result['cfd_valid']:raise RuntimeError('CFD result QA failed')
        receipt.update(status='FINISHED',elapsed_s=time.time()-started,utc=now())
        save(case/'process.json',receipt)
except Exception:
    receipt.update(status='FAILED',error=traceback.format_exc(),elapsed_s=time.time()-started,utc=now())
    save(case/'process.json',receipt);state(case,'LEGACY_SOLVER_FAILED',error=receipt['error'])
    raise
