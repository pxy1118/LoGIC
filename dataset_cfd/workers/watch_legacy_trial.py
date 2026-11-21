"""Bound resource use for one owned CFD process; exits when that attempt exits."""
from pathlib import Path
import argparse
import sys
import time
import psutil
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.common import ROOT,read,save,state,now

p=argparse.ArgumentParser();p.add_argument('--case',type=Path,required=True);a=p.parse_args()
case=a.case.resolve()
if not case.is_relative_to(ROOT/'runs'):raise ValueError('Case outside CFD runs')
r=read(case/'process.json');expected_pid=r['pid'];expected_created=r['created']
watch={'status':'MONITORING','pid':expected_pid,'created':expected_created,
       'monitoring_started_utc':now(),'peak_observed_RSS_GiB':0.,'limits':{'min_available_RAM_GiB':8,'max_RSS_GiB':40}}
strikes=0
while True:
    try:
        process=psutil.Process(expected_pid)
        if abs(process.create_time()-expected_created)>.1:break
        command=' '.join(process.cmdline())
        if not any(name in command for name in ['legacy_solve_cached.py','legacy_pilot.py']):
            raise RuntimeError('Refuse to control an unowned process')
        current=read(case/'process.json')
        if current['status']!='RUNNING':break
        rss=process.memory_info().rss/2**30;available=psutil.virtual_memory().available/2**30
        watch.update(peak_observed_RSS_GiB=max(watch['peak_observed_RSS_GiB'],rss),
                     current_RSS_GiB=rss,available_RAM_GiB=available,updated_utc=now())
        strikes=strikes+1 if available<8 or rss>40 else 0
        save(case/'resource_watch.json',watch)
        if strikes>=2:
            # Recheck identity and run state immediately before touching it.
            if abs(process.create_time()-expected_created)<.1 and read(case/'process.json')['status']=='RUNNING':
                process.terminate();process.wait(timeout=20)
                current.update(status='RESOURCE_STOP_RETAINED',utc=now())
                save(case/'process.json',current)
                state(case,'RESOURCE_STOP_RETAINED',watch=watch)
                watch['status']='RESOURCE_STOP_RETAINED';save(case/'resource_watch.json',watch)
            break
    except psutil.NoSuchProcess:break
    time.sleep(2)
if watch['status']=='MONITORING':
    watch.update(status='OWNER_EXITED_OR_FINISHED',finished_utc=now())
    save(case/'resource_watch.json',watch)
