"""Step 3: export checked CFD labels, failures, coverage and replacement requests."""
import argparse
from collections import Counter,defaultdict
import csv
import math
from src.common import read,save,digest,now
from src.batch_stages import run_root,validate_case
from pathlib import Path


def write_csv(path, rows, fields):
    temporary=path.with_name(path.name+'.tmp')
    with temporary.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    temporary.replace(path)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',required=True)
    args=parser.parse_args(argv);root=run_root(args.run);meta=read(root/'run.json')
    if digest(root/'protocol.json')!=meta['protocol_sha256']:
        raise RuntimeError('Frozen profile changed')
    inventory=read(root/'source_inventory.json')
    if digest(root/'source_inventory.json')!=meta['inventory_sha256']:
        raise RuntimeError('Frozen source inventory changed')
    cfd=read(root/'batch_cfd.json') if (root/'batch_cfd.json').exists() else {'jobs':{}}
    geometry_root=Path(cfd.get('geometry_run',str(root)))
    geometry=read(geometry_root/'batch_geometry.json') if (geometry_root/'batch_geometry.json').exists() else {'jobs':{}}
    labels=[];failures=[];groups=defaultdict(Counter)
    metrics=['K_domain_app_z_m2','ln_K_domain_app_z','pressure_gradient_z_Pa_m',
             'delta_p_domain_Pa','Q_m3_s','mass_imbalance_rel','solve_elapsed_s']
    identity=['sample_id','source_sample_id','split','topology','kind','relative_density','paired_fea','geometry_group']
    for source in inventory['samples']:
        sid=str(source['sample_id']);g=geometry['jobs'].get(sid,{});c=cfd['jobs'].get(sid,{})
        row={k:source[k] for k in identity if k!='topology'}
        row['topology']='+'.join(source['topology_family'])
        rho=source['relative_density']
        density_bin='rho<0.30' if rho<.30 else '0.30<=rho<0.40' if rho<.40 else 'rho>=0.40'
        counts=groups[(source['split'],row['topology'],source['kind'],density_bin)];counts['source_count']+=1
        counts['geometry_attempted']+=bool(g.get('attempts'))
        counts['geometry_success']+=g.get('status')=='SUCCESS'
        counts['cfd_attempted']+=bool(c.get('attempts'))
        reason=None
        if c.get('status')=='SUCCESS':
            try:
                case=Path(c['case']);geometry_case=Path(g['case'])
                validate_case(case,'cfd',meta['protocol_sha256'],source,geometry_case)
                result=read(case/'results.json')
                values={k:result[k] for k in metrics}
                if not all(math.isfinite(v) for v in values.values()):
                    raise ValueError('Nonfinite CFD label')
                labels.append({**row,**values,'tetrahedra':result['mesh']['elements'],
                    'production_valid':result['production_valid'],'protocol_sha256':meta['protocol_sha256'],
                    'case':str(case)})
                counts['cfd_success']+=1
            except (ValueError,KeyError,FileNotFoundError) as error:
                reason=('EXPORT_VALIDATION_FAILED',str(error),c.get('case',''))
        elif c.get('status')=='FAILED':
            reason=(c.get('failure_stage','CFD_FAILED'),c.get('error_last_line',''),c.get('case',''))
        elif g.get('status')=='FAILED':
            reason=(g.get('failure_stage','GEOMETRY_FAILED'),g.get('error_last_line',''),g.get('case',''))
        if reason:
            failures.append({**row,'failure_stage':reason[0],'reason':reason[1],'case':reason[2]})
            counts['failed']+=1
    output=root/'results';output.mkdir(exist_ok=True)
    write_csv(output/'labels.csv',labels,identity+metrics+['tetrahedra','production_valid','protocol_sha256','case'])
    write_csv(output/'failures.csv',failures,identity+['failure_stage','reason','case'])
    requests=[{**{k:r[k] for k in identity},'new_sample_id':'','reason':r['failure_stage']} for r in failures]
    write_csv(output/'replacement_requests.csv',requests,identity+['new_sample_id','reason'])
    coverage=[dict(split=key[0],topology=key[1],kind=key[2],density_bin=key[3],
                   **{field:value.get(field,0) for field in ['source_count','geometry_attempted','geometry_success',
                       'cfd_attempted','cfd_success','failed']}) for key,value in sorted(groups.items())]
    write_csv(output/'coverage.csv',coverage,['split','topology','kind','density_bin','source_count','geometry_attempted',
             'geometry_success','cfd_attempted','cfd_success','failed'])
    summary={'updated_utc':now(),'run':args.run,'source_count':len(inventory['samples']),
             'collector_source_sha256':digest(Path(__file__)),
             'valid_cfd_count':len(labels),'valid_split_counts':dict(Counter(r['split'] for r in labels)),
             'valid_paired_fea_count':sum(r['paired_fea'] for r in labels),
             'recorded_failures':len(failures),'geometry_status_counts':dict(Counter(r['status'] for r in geometry['jobs'].values())),
             'cfd_status_counts':dict(Counter(r['status'] for r in cfd['jobs'].values())),
             'production_valid_count':sum(r['production_valid'] for r in labels),
             'replacement_policy':'requests only; no IDs overwritten or new geometries generated',
             'note':'Counts describe attempted/accepted simulations; unavailable labels are never zero-filled'}
    save(output/'summary.json',summary);print(summary)


if __name__=='__main__':main()
