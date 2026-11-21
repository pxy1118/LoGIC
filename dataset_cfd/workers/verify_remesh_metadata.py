"""After an owned attempt exits, correct descriptive tags with raw backups."""
from pathlib import Path
import argparse
import sys
import time
import shutil
import psutil
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.common import ROOT,read,save,now
p=argparse.ArgumentParser();p.add_argument('--case',type=Path,required=True);a=p.parse_args()
case=a.case.resolve()
if not case.is_relative_to(ROOT/'runs'):raise ValueError('Case outside CFD runs')
owner=read(case/'process.json')
while True:
    try:
        proc=psutil.Process(owner['pid'])
        if abs(proc.create_time()-owner['created'])>.1:break
    except psutil.NoSuchProcess:break
    time.sleep(2)
cfg=read(case/'protocol.json');sequence=cfg['geometry']['surface_remesh_sequence']
mode='external_3matic_'+'_'.join(sequence)+'_then_COMSOL_planar' if sequence else 'COMSOL_planar_faces_only'
changes=[]
for name in ['cfd_mesh_report.json','results.json']:
    path=case/name
    if not path.exists():continue
    data=read(path);mesh=data['mesh'] if name=='results.json' else data
    if mesh.get('surface_remesh_mode')==mode:continue
    backup=case/'metadata_originals';backup.mkdir(exist_ok=True)
    if not (backup/name).exists():shutil.copy2(path,backup/name)
    changes.append({'file':name,'old':mesh.get('surface_remesh_mode'),'new':mode})
    mesh['surface_remesh_mode']=mode;mesh['surface_remesh_sequence']=sequence
    save(path,data)
save(case/'metadata_verification.json',{'utc':now(),'changes':changes,
     'numerical_values_changed':False,'note':'Correct descriptive remesh tag from executed prefix protocol; original JSON files retained'})
