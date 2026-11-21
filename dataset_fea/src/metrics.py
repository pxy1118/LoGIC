from __future__ import annotations
import csv
import gzip
import json
import numpy as np
from .common import read,save,digest,state


def curve_metrics(strain,stress,mass_g,cfg):
    """No extrapolation, silent zero padding, smoothing, or alternate yield definitions."""
    x=np.asarray(strain,dtype=float);y=np.asarray(stress,dtype=float)
    if not np.isfinite(mass_g) or mass_g<=0:raise ValueError('Positive finite initial mass required')
    if len(x)!=len(y) or len(x)<3 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Invalid curve')
    if np.any(np.diff(x)<-1e-10): raise ValueError('Compression curve contains unloading/reversed strain')
    # Only numerically repeated prescribed-displacement points, preserving chronological order.
    keep=np.r_[True,np.diff(x)>1e-12];x=x[keep];y=y[keep]
    target=cfg['loading']['max_strain'];tol=target*cfg['quality']['end_displacement_fraction']
    if abs(x[0])>1e-8 or x[-1]<target-tol or x[-1]>target+tol:
        raise ValueError('Curve does not reach the required strain endpoint')
    # Round-off-only endpoint snapping is explicit and bounded, not extrapolation.
    endpoint_adjustment=float(target-x[-1]);x=x.copy();x[-1]=target
    q=cfg['quality'];lo,hi=q['elastic_window'];mask=(x>=lo)&(x<=hi)
    E,b,r2=None,None,None;reasons=[];sensitivity=None
    if mask.sum()>=q['elastic_min_points']:
        slope,intercept=np.polyfit(x[mask],y[mask],1)
        residual=y[mask]-(slope*x[mask]+intercept)
        r2=float(1-np.sum(residual**2)/max(np.sum((y[mask]-y[mask].mean())**2),1e-30))
        halves=[mask&(x<=(lo+hi)/2),mask&(x>=(lo+hi)/2)]
        slopes=[np.polyfit(x[m],y[m],1)[0] for m in halves if m.sum()>=3]
        sensitivity=float(max(abs(s-slope) for s in slopes)/max(abs(slope),1e-30)) if slopes else None
        if slope>0 and r2>=q['elastic_min_r2'] and sensitivity is not None and sensitivity<=q['elastic_slope_change_fraction']:
            E,b=float(slope),float(intercept)
        else: reasons.append('elastic_window_not_linear_or_stable')
    else: reasons.append('insufficient_elastic_points')
    yield_stress=yield_strain=None;yield_reason='invalid_elastic_modulus' if E is None else 'not_observed_by_5pct'
    if E is not None:
        diff=y-(E*(x-.002)+b)
        for i in range(1,len(x)):
            if x[i]<max(.002,hi) or not (diff[i-1]>0>=diff[i]): continue
            # Require persistence for at least 0.0001 strain to reject isolated oscillatory crossings.
            end=np.searchsorted(x,x[i]+.0001)
            if end>=len(x) or np.any(diff[i:end+1]>0): continue
            a=diff[i-1]/(diff[i-1]-diff[i]);yield_strain=float(x[i-1]+a*(x[i]-x[i-1]))
            yield_stress=float(y[i-1]+a*(y[i]-y[i-1]));yield_reason='observed';break
    strain_grid=np.linspace(0,target,1001);stress_grid=np.interp(strain_grid,x,y)
    W={}
    for percent in range(1,6):
        limit=percent*.01;sel=x<limit
        xx=np.r_[x[sel],limit]; yy=np.r_[y[sel],np.interp(limit,x,y)]
        W['W%d_MJ_m3'%percent]=float(np.trapezoid(yy,xx))
    total_J=W['W5_MJ_m3']*cfg['geometry']['size_mm']**3/1000
    result=dict(schema_version=1,definition='engineering compression, fixed envelope, signed force',
       elastic_modulus_MPa=E,elastic_fit=dict(window=[lo,hi],intercept_MPa=b,r2=r2,points=int(mask.sum()),slope_sensitivity=sensitivity),
       yield_strength_MPa=yield_stress,yield_strain=yield_strain,yield_status=yield_reason,
       yield_method='first persistent 0.2 percent offset intersection with fitted intercept',
       **W,total_work_J=total_J,SEA_J_g=total_J/mass_g,
       peak_stress_MPa=float(y.max()),peak_strain=float(x[np.argmax(y)]),
       endpoint_adjustment_strain=endpoint_adjustment,label_warnings=reasons,
       valid=dict(curve=True,elastic=E is not None,yield_strength=yield_stress is not None,energy=True))
    return result,strain_grid,stress_grid


def energy_quality(energy,cfg):
    whole=[v for k,v in energy.items() if 'ETOTAL' in v and 'ALLIE' in v]
    solids=[v for k,v in energy.items() if 'SOLID' in k.upper() and 'ALLIE' in v]
    if len(whole)!=1 or len(solids)!=1: raise ValueError('Expected distinct whole-model and SOLID energy histories')
    def aligned(region,key,t):
        if key not in region: raise ValueError('Missing energy variable '+key)
        a=np.array(region[key],float)
        if not np.isfinite(a).all() or a[0,0]>t[0]+1e-10 or a[-1,0]<t[-1]-1e-10:
            raise ValueError('Incomplete energy history '+key)
        return np.interp(t,a[:,0],a[:,1])
    whole,solid=whole[0],solids[0]
    t=np.array(solid['ALLIE'])[:,0];ie=aligned(solid,'ALLIE',t)
    active=np.abs(ie)>max(1e-12,np.max(np.abs(ie))*.001)
    if active.sum()<10: raise ValueError('Insufficient active energy history')
    ratios={}
    for key in ('ALLKE','ALLAE'):
        values=np.abs(aligned(solid,key,t)[active])/np.abs(ie[active])
        ratios[key+'_over_ALLIE_p95']=float(np.quantile(values,.95))
        ratios[key+'_over_ALLIE_max']=float(values.max())
        ratios[key+'_fraction_above_5pct']=float(np.mean(values>.05))
    total=aligned(whole,'ETOTAL',t);work=aligned(whole,'ALLWK',t)
    ratios['energy_balance_fraction']=float(np.max(np.abs(total-total[0]))/max(np.max(np.abs(work)),1e-12))
    masses=[np.array(v['DMASS'],float)[:,1] for v in energy.values() if 'DMASS' in v]
    if not masses: raise ValueError('Missing DMASS history')
    ratios['maximum_mass_change_percent']=float(max(np.max(np.abs(v)) for v in masses))
    q=cfg['quality'];failures=[]
    for field,limit in [('ALLKE_over_ALLIE_p95','kinetic_ratio_p95'),('ALLAE_over_ALLIE_p95','artificial_ratio_p95'),
                        ('energy_balance_fraction','energy_balance_fraction'),('maximum_mass_change_percent','maximum_mass_change_percent')]:
        if ratios[field]>q[limit]:failures.append(field)
    return dict(passed=not failures,failures=failures,**ratios)


def postprocess(case,cfg):
    extraction=read(case/'extraction.json')
    with gzip.open(case/'raw_history.csv.gz','rt',encoding='utf-8') as stream:
        rows=list(csv.DictReader(stream))
    a={k:np.array([float(r[k]) for r in rows]) for k in rows[0]}
    if not np.all(np.diff(a['time_s'])>0): raise ValueError('Non-increasing history times')
    x=(a['bottom_u3_mm']-a['top_u3_mm'])/cfg['geometry']['size_mm']
    y=a['bottom_rf3_N']/cfg['geometry']['size_mm']**2
    mesh=read(case/'mesh_quality.json')
    result,grid,stress=curve_metrics(x,y,mesh['mass_g'],cfg)
    with gzip.open(case/'energy_history.json.gz','rt',encoding='utf-8') as stream:
        energy=json.load(stream)
    eq=energy_quality(energy,cfg)
    scale=max(float(np.max(np.abs(a['bottom_rf3_N']))),1e-12)
    instantaneous=np.maximum(np.abs(a['bottom_rf3_N']),np.abs(a['top_rf3_N']))
    active=instantaneous>.01*scale
    imbalance=np.abs(a['bottom_rf3_N']+a['top_rf3_N'])/np.maximum(instantaneous,1e-12)
    imbalance_p95=float(np.quantile(imbalance[active],.95)) if active.any() else float(1)
    failures=list(eq['failures'])
    if not extraction['solver_completed']:failures.append('solver_incomplete')
    if not result['valid']['elastic']:failures.append('elastic_label_invalid')
    if imbalance_p95>cfg['quality']['force_imbalance_p95']:failures.append('force_imbalance')
    if np.min(y)<-max(.1,float(y.max())*.01):failures.append('negative_compression_force')
    T=read(case/'analysis.json')['step_time_s']
    if abs(extraction['last_history_time_s']/T-1)>1e-6:failures.append('incomplete_time')
    if len(rows)<cfg['solver']['history_intervals']*.95:failures.append('sparse_history')
    quality=dict(passed=not failures,failures=failures,energy=eq,force_imbalance_p95=imbalance_p95,
                 scope='numerical checks; mesh/rate/material validation is a separate protocol requirement')
    if (case/'voxel_fea.npz').exists():
        from .package import canonical_group
        quality['geometry_group']=canonical_group(np.load(case/'voxel_fea.npz')['voxel'])
    save(case/'metrics.json',result);save(case/'quality.json',quality)
    np.savez_compressed(case/'curve.npz',strain=grid,stress_MPa=stress)
    with (case/'stress_strain.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.writer(f);writer.writerow(['strain','stress_MPa']);writer.writerows(zip(grid,stress))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(6,4));ax.plot(x*100,y,lw=1,label='Raw FEA')
    ax.set(xlabel='Engineering compression strain (%)',ylabel='Engineering stress (MPa)',xlim=(0,5))
    ax.grid(alpha=.2);ax.set_title(case.name+' | '+('PASS' if quality['passed'] else 'REVIEW'))
    fig.tight_layout();fig.savefig(case/'curve.png',dpi=160);plt.close(fig)
    names=['raw_history.csv.gz','energy_history.json.gz','field_diagnostics.json','sparse_displacements.json.gz',
           'extraction.json','metrics.json','quality.json','curve.npz','stress_strain.csv','curve.png','analysis.json']
    if (case/'solver_execution.json').exists():names.append('solver_execution.json')
    save(case/'export_manifest.json',dict(hashes={name:digest(case/name) for name in names}))
    state(case,'VALIDATED' if quality['passed'] else 'NUMERICAL_REVIEW',failures=failures)
    return quality
