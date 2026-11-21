from __future__ import annotations
import numpy as np
import math
from .common import save,digest,read,state
from .mesh import read_tet10


def mass_scaling_keyword(cfg):
    solver=cfg['solver'];dt=solver.get('mass_scaling_dt_s')
    if dt is None:return None
    if isinstance(dt,bool) or not isinstance(dt,(int,float)) or not math.isfinite(dt) or dt<=0:
        raise ValueError('Positive finite mass scaling target required')
    mode=solver.get('mass_scaling_mode','fixed')
    if mode=='fixed':return '*Fixed Mass Scaling, elset=SOLID, type=BELOW MIN, dt=%.12g'%dt
    if mode!='variable':raise ValueError('Mass scaling mode must be fixed or variable')
    frequency=solver.get('mass_scaling_frequency',100)
    if isinstance(frequency,bool) or not isinstance(frequency,int) or frequency<1:
        raise ValueError('Positive integer mass scaling frequency required')
    return '*Variable Mass Scaling, elset=SOLID, type=BELOW MIN, dt=%.12g, frequency=%d'%(dt,frequency)


def write_deck(case,cfg,speed,step_time_override=None):
    if speed is None or speed<=0: raise ValueError('Loading speed is not selected; supply a calibration speed')
    damage=not cfg['material'].get('validation_elastic_only',False)
    if damage and cfg['loading'].get('platen_contact','general')!='general':
        raise ValueError('Material failure requires general contact with an eroding specimen surface')
    if (case/'analysis.inp').exists():
        old=read(case/'analysis.json')
        if digest(case/'analysis.inp')!=old['input_sha256']: raise RuntimeError('Input deck changed')
        if old['mean_speed_mm_s']!=speed or old.get('step_time_override')!=step_time_override:
            raise RuntimeError('Use a new case/run for a changed loading protocol')
        return old
    scaling=mass_scaling_keyword(cfg)
    labels,coords,eids,conn=read_tet10(case/'volume.inp')
    size=cfg['geometry']['size_mm']; travel=size*cfg['loading']['max_strain']
    T=travel/speed if step_time_override is None else step_time_override
    mat=cfg['material']; nodes=[(int(k),*xyz) for k,xyz in zip(labels,coords)]
    next_node=int(labels.max())+1; next_element=int(eids.max())+1
    half=cfg['loading']['platen_size_mm']/2; mid=size/2
    plates=[]
    for z,reverse in [(float(coords[:,2].min()),False),(float(coords[:,2].max()),True)]:
        ids=[]
        for x,y in [(-half,-half),(half,-half),(half,half),(-half,half)]:
            nodes.append((next_node,mid+x,mid+y,z));ids.append(next_node);next_node+=1
        if reverse: ids.reverse()
        plates.append((next_element,ids));next_element+=1
    bottom_rp,top_rp=next_node,next_node+1
    nodes.extend([(bottom_rp,mid,mid,float(coords[:,2].min())),(top_rp,mid,mid,float(coords[:,2].max()))])
    path=case/'analysis.inp'
    with path.open('w',encoding='ascii',newline='\n') as f:
        def line(s): f.write(s+'\n')
        line('*Heading\nTPMS dataset; units N mm tonne s MPa\n*Preprint, echo=NO, model=NO, history=NO, contact=NO\n*Node')
        for row in nodes: line('%d, %.12g, %.12g, %.12g'%row)
        line('*Element, type='+cfg['mesh']['element_type']+', elset=SOLID')
        for eid,indices in zip(eids,conn): line(', '.join(map(str,[eid,*labels[indices]])))
        for name,(eid,ids),rp in zip(('BOT','TOP'),plates,(bottom_rp,top_rp)):
            line('*Element, type=R3D4, elset=PLATE_'+name); line(', '.join(map(str,[eid,*ids])))
            line('*Nset, nset='+name+'_RP');line(str(rp))
            line('*Rigid Body, ref node='+name+'_RP, elset=PLATE_'+name)
            line('*Surface, type=ELEMENT, name=SURF_'+name+'\nPLATE_'+name+', SPOS')
        line('*Surface, type=ELEMENT, name=SPECIMEN\nSOLID,'+(' ERODING' if damage else ''))
        line('*Material, name=TI\n*Density');line(str(mat['density_tonne_mm3']))
        line('*Elastic');line('%g, %g'%(mat['E_MPa'],mat['nu']))
        if not mat.get('validation_elastic_only',False):
            line('*Plastic')
            for stress,eps in mat['plastic']: line('%g, %g'%(stress,eps))
            line('*Damage Initiation, criterion=JOHNSON COOK');line(', '.join(map(str,mat['johnson_cook_damage'])))
            line('*Damage Evolution, type=DISPLACEMENT');line(str(mat['damage_displacement_mm']))
        # C3D10M does not support distortion control (Abaqus datacheck warning).
        line('*Section Controls, name=SC, element deletion=YES')
        line('*Solid Section, elset=SOLID, material=TI, controls=SC\n,')
        line('*Surface Interaction, name=CONTACT\n*Surface Behavior, pressure-overclosure=HARD\n*Friction')
        line(str(cfg['loading']['friction']))
        line('*Boundary\nBOT_RP, ENCASTRE\nTOP_RP, 1, 2, 0.\nTOP_RP, 4, 6, 0.')
        line('*Amplitude, name=RAMP, definition=SMOOTH STEP\n0., 0., %.12g, 1.'%T)
        line('*Step, name=Compression, nlgeom=YES\n*Dynamic, Explicit\n, %.12g'%T)
        line('*Bulk Viscosity\n0.06, 1.2')
        dt=cfg['solver']['mass_scaling_dt_s']
        if scaling:line(scaling)
        line('*Contact, op=NEW')
        if damage:
            # ALL EXTERIOR alone does not include faces exposed by element deletion.
            line('*Contact Inclusions\nSPECIMEN, SPECIMEN\nSPECIMEN, SURF_BOT\nSPECIMEN, SURF_TOP')
        else:line('*Contact Inclusions, ALL EXTERIOR')
        line('*Contact Property Assignment\n, , CONTACT')
        penalty=cfg['loading'].get('contact_penalty_scale',1.)
        if penalty<=0: raise ValueError('Positive contact penalty scale required')
        line('*Contact Controls Assignment, type=SCALE PENALTY\n, , %.12g'%penalty)
        if cfg['loading'].get('platen_contact','general')=='kinematic':
            line('*Contact Pair, interaction=CONTACT, mechanical constraint=KINEMATIC\nSPECIMEN, SURF_BOT\nSPECIMEN, SURF_TOP')
        line('*Boundary, amplitude=RAMP\nTOP_RP, 3, 3, %.12g'%-travel)
        line('*Output, field, number interval=%d'%cfg['solver']['field_intervals'])
        line('*Node Output\nU\n*Element Output, elset=SOLID\nS, PEEQ, STATUS')
        interval=T/cfg['solver']['history_intervals']
        for name in ('TOP','BOT'):
            line('*Output, history, time interval=%.12g'%interval)
            line('*Node Output, nset='+name+'_RP\nU3, RF3')
        line('*Output, history, time interval=%.12g'%interval)
        line('*Energy Output\nALLIE, ALLKE, ALLAE, ALLVD, ALLPD, ALLDMD, ALLFD, ALLWK, ALLPW, ALLCW, ALLMW, ETOTAL')
        line('*Energy Output, elset=SOLID\nALLIE, ALLKE, ALLAE, ALLPD, ALLDMD')
        line('*Incrementation Output\nDT, DMASS\n*End Step')
    record=dict(input_sha256=digest(path),config=cfg,mean_speed_mm_s=speed,step_time_s=T,
                step_time_override=step_time_override,actual_mean_speed_mm_s=travel/T,
                peak_speed_mm_s=1.875*travel/T,travel_mm=travel,bottom_rp=bottom_rp,top_rp=top_rp,
                initial_height_mm=float(np.ptp(coords[:,2])),nodes=len(labels),elements=len(eids),
                mass_scaling_dt_s=dt,material_scope=mat['calibration_status'])
    save(case/'analysis.json',record);state(case,'PREPARED')
    return record
