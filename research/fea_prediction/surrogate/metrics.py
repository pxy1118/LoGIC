"""Frozen FEA curve metric definition; evaluated on the 1001-point prediction grid.
Source: dataset_fea/src/metrics.py
Source SHA256: acfe68b5c7f83c7a7bd02da1ea9dc389db5b28ef2db4b096e8b217e7a11ba25a
"""
import numpy as np

QUALITY_DEFAULTS = {'bounds_tolerance_mm': 0.01, 'volume_drift_fraction': 0.01, 'elastic_window': [0.0005, 0.0025], 'elastic_min_points': 20, 'elastic_min_r2': 0.995, 'elastic_slope_change_fraction': 0.1, 'kinetic_ratio_p95': 0.05, 'artificial_ratio_p95': 0.05, 'energy_balance_fraction': 0.02, 'force_imbalance_p95': 0.05, 'maximum_mass_change_percent': 5, 'end_displacement_fraction': 0.001}

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
