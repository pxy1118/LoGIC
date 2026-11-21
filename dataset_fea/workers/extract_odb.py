"""Abaqus Python worker. Export small histories and diagnostics before ODB cleanup."""
import csv
import gzip
import json
from pathlib import Path
import sys
import traceback
from odbAccess import openOdb
from abaqusConstants import DOUBLE_PRECISION


def field_data(value):
    return value.dataDouble if value.precision == DOUBLE_PRECISION else value.data


def write_json(path,data):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,indent=2,allow_nan=False),encoding='utf-8')
    tmp.replace(path)


def run(case):
    cfg=json.loads((case/'analysis.json').read_text(encoding='utf-8'))
    odb_path=case/'analysis.odb'
    if (case/'analysis.lck').exists(): raise RuntimeError('ODB remains locked')
    odb=openOdb(path=str(odb_path),readOnly=True)
    try:
        step=odb.steps['Compression']
        regions=step.historyRegions
        def node_history(label):
            matches=[r for name,r in regions.items() if name.startswith('Node ') and
                     name.rsplit('.',1)[-1]==str(label) and 'RF3' in r.historyOutputs]
            if len(matches)!=1: raise RuntimeError('Ambiguous/missing reference node history %s'%label)
            return matches[0]
        top=node_history(cfg['top_rp']);bot=node_history(cfg['bottom_rp'])
        series=[top.historyOutputs[k].data for k in ('U3','RF3')]
        series += [bot.historyOutputs[k].data for k in ('U3','RF3')]
        if len(set(len(s) for s in series))!=1 or len(series[0])<2:
            raise RuntimeError('Histories are incomplete or misaligned')
        temp=case/'raw_history.csv.gz.tmp'
        with gzip.open(str(temp),'wt',newline='',encoding='utf-8') as stream:
            w=csv.writer(stream);w.writerow(['time_s','top_u3_mm','top_rf3_N','bottom_u3_mm','bottom_rf3_N'])
            for rows in zip(*series):
                times=[r[0] for r in rows]
                if max(times)-min(times)>max(1e-12,cfg['step_time_s']*1e-7):
                    raise RuntimeError('History timestamps differ')
                w.writerow([times[0]]+[r[1] for r in rows])
        temp.replace(case/'raw_history.csv.gz')
        energy={}
        for name,region in regions.items():
            values={key:list(h.data) for key,h in region.historyOutputs.items()
                    if key.startswith('ALL') or key in ('ETOTAL','DT','DMASS')}
            if values: energy[name]=values
        temp=case/'energy_history.json.gz.tmp'
        with gzip.open(str(temp),'wt',encoding='utf-8') as stream:
            json.dump(energy,stream,allow_nan=False)
        temp.replace(case/'energy_history.json.gz')
        if not any('ALLKE' in r and 'ALLIE' in r for r in energy.values()):
            raise RuntimeError('Required energy histories missing')
        fields=[]
        for frame in step.frames:
            entry=dict(time_s=frame.frameValue)
            for key in ('PEEQ','STATUS','SDEG','JCCRT','DMICRT','STATUSMP','EMSF'):
                if key in frame.fieldOutputs:
                    vals=[]
                    for v in frame.fieldOutputs[key].values:
                        data=field_data(v)
                        vals.append(float(max(data) if isinstance(data,(tuple,list)) else data))
                    if vals:
                        entry[key+'_min']=min(vals);entry[key+'_max']=max(vals)
                        entry[key+'_count']=len(vals)
                        if key=='STATUS': entry['deleted_elements']=sum(v<.5 for v in vals)
                        if key=='PEEQ': entry['plastic_points']=sum(v>1e-8 for v in vals)
                        if key=='SDEG':
                            entry['damaged_points']=sum(v>1e-8 for v in vals)
                            entry['fully_degraded_points']=sum(v>=1-1e-6 for v in vals)
                        if key=='JCCRT':entry['damage_initiated_points']=sum(v>=1-1e-6 for v in vals)
                        if key=='DMICRT':
                            entry['damage_initiated_points']=sum(v>=1-1e-6 for v in vals)
                            entry['damage_criterion_components']=list(frame.fieldOutputs[key].componentLabels)
                        if key=='STATUSMP':entry['failed_material_points']=sum(v<.5 for v in vals)
            fields.append(entry)
        write_json(case/'field_diagnostics.json',fields)
        # Sparse displacements support future failure review without keeping ODB.
        sparse=[]
        for index in sorted(set([0,len(step.frames)//2,len(step.frames)-1])):
            frame=step.frames[index]
            if 'U' in frame.fieldOutputs:
                sparse.append(dict(time_s=frame.frameValue,displacements=[
                    [v.nodeLabel,*list(field_data(v))] for v in frame.fieldOutputs['U'].values]))
        temp=case/'sparse_displacements.json.gz.tmp'
        with gzip.open(str(temp),'wt',encoding='utf-8') as stream:
            json.dump(sparse,stream,allow_nan=False)
        temp.replace(case/'sparse_displacements.json.gz')
        complete='THE ANALYSIS HAS COMPLETED SUCCESSFULLY' in (case/'analysis.sta').read_text(errors='replace')
        write_json(case/'extraction.json',dict(status='EXTRACTED',solver_completed=complete,
            samples=len(series[0]),last_history_time_s=series[0][-1][0],last_frame_time_s=step.frames[-1].frameValue,
            energy_regions=list(energy),frames=len(step.frames)))
    finally:
        odb.close()


if __name__=='__main__':
    folder=Path(sys.argv[-1]).resolve()
    try: run(folder)
    except Exception:
        write_json(folder/'extraction_error.json',dict(error=traceback.format_exc()))
        raise
