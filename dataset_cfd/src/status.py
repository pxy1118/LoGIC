from pathlib import Path
import psutil
import shutil
from .common import ROOT, read


def status(run):
    folder=ROOT/'runs'/run
    samples=[]
    for case in sorted((folder/'samples').glob('*'),key=lambda p:int(p.name)):
        d=read(case/'state.json') if (case/'state.json').exists() else {'stage':'PENDING'}
        item={'sample_id':int(case.name),'stage':d['stage'],'updated_utc':d.get('updated_utc')}
        if (case/'fluid_report.json').exists():
            r=read(case/'fluid_report.json');item.update(production_geometry_valid=r.get('production_geometry_valid'),
                   nonmanifold_edges=r.get('nonmanifold_edges'),volume_closure_rel=r.get('volume_closure_rel'))
        if (case/'results.json').exists():
            r=read(case/'results.json');item['results']={k:r.get(k) for k in
                ['K_app_z_m2','mass_imbalance_rel','cfd_valid','production_valid']}
        if (case/'solver_process.json').exists():
            r=read(case/'solver_process.json')
            try:
                p=psutil.Process(r['pid'])
                if r['status']=='RUNNING' and abs(p.create_time()-r['created'])<.1:
                    item['live_solver']={'pid':p.pid,'rss_GiB':round(p.memory_info().rss/2**30,3),
                        'cpu_seconds':round(sum(p.cpu_times()[:2]),1)}
            except (psutil.Error,KeyError):pass
        samples.append(item)
        for variant in sorted((case/'variants').glob('*')):
            entry={'sample_id':int(case.name),'variant':variant.name}
            if (variant/'state.json').exists():entry.update(stage=read(variant/'state.json')['stage'])
            if (variant/'results.json').exists():
                r=read(variant/'results.json');entry['results']={k:r.get(k) for k in
                    ['K_app_z_m2','mass_imbalance_rel','cfd_valid','production_valid','mesh_quality_review_required']}
            if (variant/'solver_process.json').exists():
                r=read(variant/'solver_process.json')
                try:
                    proc=psutil.Process(r['pid'])
                    if r['status']=='RUNNING' and abs(proc.create_time()-r['created'])<.1:
                        entry['live_solver']={'pid':proc.pid,'rss_GiB':round(proc.memory_info().rss/2**30,3),
                             'cpu_seconds':round(sum(proc.cpu_times()[:2]),1)}
                except (psutil.Error,KeyError):pass
            samples.append(entry)
    benchmarks={p.parent.name:read(p) for p in (folder/'benchmarks').glob('*/results.json')}
    return {'run':run,'samples':samples,'benchmarks':benchmarks,
            'mesh_study':read(folder/'mesh_study.json') if (folder/'mesh_study.json').exists() else None,
            'mesh_study_revisions':{p.stem:read(p) for p in folder.glob('mesh_study_*.json')},
            'pilot_preparation':read(folder/'pilot_preparation.json') if (folder/'pilot_preparation.json').exists() else None,
            'free_disk_GiB':round(shutil.disk_usage(ROOT).free/2**30,2),
            'available_RAM_GiB':round(psutil.virtual_memory().available/2**30,2)}
