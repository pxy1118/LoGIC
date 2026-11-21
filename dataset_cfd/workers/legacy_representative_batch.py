"""Run a small train-only pilot serially, preserving every native case.

This is deliberately limited to the preselected development geometries;
it does not launch the 500-case production queue.
"""
from pathlib import Path
import argparse
import os
import shutil
import subprocess
import sys
import time
import traceback
import psutil

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.common import ROOT,read,save,digest,lock,now


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples',type=int,nargs='+',required=True)
    parser.add_argument('--run',required=True)
    parser.add_argument('--profile',type=Path,required=True)
    args=parser.parse_args()
    if not args.run.replace('_','').replace('-','').isalnum():
        raise ValueError('Invalid run name')
    if len(set(args.samples))!=len(args.samples) or not 1<=len(args.samples)<=12:
        raise ValueError('Select 1–12 distinct pilot geometries')
    inventory=read(ROOT/'plan/source_inventory_20261107.json')
    allowed=set(inventory['pilot_selection']['sample_ids'])
    selected=[s for s in inventory['samples'] if s['sample_id'] in args.samples]
    if (not set(args.samples)<=allowed or len(selected)!=len(args.samples)
            or any(s['split']!='train' for s in selected)):
        raise ValueError('Only preselected training pilot geometries are allowed')
    run=ROOT/'runs'/args.run
    if (run/'batch.json').exists() or (run/'samples').exists():
        raise RuntimeError('Existing batch preserved; choose a new run')
    run.mkdir(parents=True,exist_ok=True)
    shutil.copy2(args.profile.resolve(),run/'batch_protocol.json')
    paths=[*sorted((ROOT/'src').glob('*.py')),Path(__file__),
           *[ROOT/'workers'/name for name in ['legacy_pilot.py','legacy_fluid_blender.py',
             'legacy_remesh_3matic.py','watch_legacy_trial.py','verify_legacy_physics.py']]]
    hashes={str(p.relative_to(ROOT)):digest(p) for p in paths}
    snapshot=run/'batch_executed_source';snapshot.mkdir()
    for path in paths:shutil.copy2(path,snapshot/path.name)
    save(snapshot/'manifest.json',hashes)
    protocol_hash=digest(run/'batch_protocol.json')
    receipt={'status':'RUNNING','scope':'TRAINING_REPRESENTATIVE_PILOT_NOT_PRODUCTION',
             'pid':os.getpid(),'created':psutil.Process().create_time(),'started_utc':now(),
             'sample_ids':args.samples,'selected_metadata':[
                 {k:s[k] for k in ['sample_id','split','kind','topology_family','relative_density']}
                 for s in selected],
             'protocol_sha256':protocol_hash,'source_inventory_sha256':digest(ROOT/'plan/source_inventory_20261107.json'),
             'source_manifest_sha256':digest(snapshot/'manifest.json'),'cases':[],'active_sample':None}
    save(run/'batch.json',receipt)
    env=os.environ.copy();env['PYTHONIOENCODING']='utf-8'
    try:
        with lock(run/'.batch.lock'):
            for sample in args.samples:
                if digest(run/'batch_protocol.json')!=protocol_hash or any(digest(p)!=hashes[str(p.relative_to(ROOT))] for p in paths):
                    raise RuntimeError('Code or profile changed during the batch; create a new run')
                receipt.update(active_sample=sample,updated_utc=now());save(run/'batch.json',receipt)
                case=run/'samples'/str(sample)
                command=[sys.executable,str(ROOT/'workers/legacy_pilot.py'),'--sample',str(sample),
                         '--run',args.run,'--profile',str(run/'batch_protocol.json')]
                with (run/f'sample_{sample}.stdout.log').open('wb') as out,(run/f'sample_{sample}.stderr.log').open('wb') as err:
                    child=subprocess.Popen(command,cwd=ROOT.parent,env=env,stdout=out,stderr=err,
                                           creationflags=subprocess.CREATE_NO_WINDOW)
                    receipt.update(active_pid=child.pid,active_created=psutil.Process(child.pid).create_time())
                    save(run/'batch.json',receipt)
                    watcher=None
                    with (run/f'sample_{sample}.watch.log').open('wb') as watch_log:
                        while child.poll() is None:
                            if watcher is None and (case/'process.json').exists():
                                owner=read(case/'process.json')
                                if owner['pid']!=child.pid or abs(owner['created']-receipt['active_created'])>.1:
                                    raise RuntimeError('Case process receipt does not match launched owner')
                                watcher=subprocess.Popen([sys.executable,str(ROOT/'workers/watch_legacy_trial.py'),
                                    '--case',str(case)],cwd=ROOT.parent,env=env,stdout=watch_log,stderr=watch_log,
                                    creationflags=subprocess.CREATE_NO_WINDOW)
                            time.sleep(1)
                result={'sample_id':sample,'exit_code':child.returncode,'finished_utc':now(),
                        'cfd_valid':False,'production_valid':False}
                if (case/'results.json').exists() and child.returncode==0:
                    with (run/f'sample_{sample}.verification.log').open('wb') as log:
                        audit=subprocess.run([sys.executable,str(ROOT/'workers/verify_legacy_physics.py'),
                            '--case',str(case)],cwd=ROOT.parent,env=env,stdout=log,stderr=log,
                            creationflags=subprocess.CREATE_NO_WINDOW)
                    r=read(case/'results.json')
                    result.update(cfd_valid=bool(r['cfd_valid'] and audit.returncode==0),
                                  physics_verification_passed=audit.returncode==0,
                                  K_domain_app_z_m2=r['K_domain_app_z_m2'],
                                  mass_imbalance_rel=r['mass_imbalance_rel'])
                elif (case/'process.json').exists():
                    result['case_status']=read(case/'process.json')['status']
                receipt['cases'].append(result);save(run/'batch.json',receipt)
                # A timed-out native process is retained by the pilot. Do not
                # launch another geometry job while that owner is still live.
                if (case/'native_process.json').exists():
                    native=read(case/'native_process.json')
                    if native.get('status')=='RUNNING':
                        try:
                            p=psutil.Process(native['pid'])
                            if abs(p.create_time()-native['created'])<.1:
                                raise RuntimeError('Retained native process is still live; batch stopped')
                        except psutil.NoSuchProcess:pass
            receipt.update(status='FINISHED',active_sample=None,finished_utc=now(),
                           valid_case_count=sum(r['cfd_valid'] for r in receipt['cases']))
            save(run/'batch.json',receipt)
    except BaseException:
        receipt.update(status='STOPPED_RETAINED',error=traceback.format_exc(),updated_utc=now())
        save(run/'batch.json',receipt)
        raise


if __name__=='__main__':main()
