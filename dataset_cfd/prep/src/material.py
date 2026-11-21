"""Check the actual Abaqus damage formula; a source citation alone is not validation."""
from __future__ import annotations
import math
from .common import object_hash


def audit_material(cfg):
    mat=cfg['material'];reasons=[]
    if mat.get('validation_elastic_only'):reasons.append('elastic software fixture is excluded from production')
    if mat['E_MPa']<=0 or not -1<mat['nu']<.5 or mat['density_tonne_mm3']<=0:
        reasons.append('invalid elastic constants or density')
    plastic=mat['plastic']
    if plastic[0][1]!=0 or any(b[1]<=a[1] or b[0]<a[0] for a,b in zip(plastic,plastic[1:])):
        reasons.append('plastic table must start at zero plastic strain and increase monotonically')
    d1,d2,d3,d4,d5,tm,tt,rate=mat['johnson_cook_damage']
    strains={name:d1+d2*math.exp(-d3*eta) for name,eta in
             [('uniaxial_compression',-1/3),('shear',0),('uniaxial_tension',1/3)]}
    if min(strains.values())<=0:reasons.append('nonpositive damage initiation strain in a basic stress state')
    if tm<=tt or rate<=0:reasons.append('invalid Johnson-Cook temperature or reference strain rate')
    if mat.get('calibration_status')!='VALIDATED':reasons.append('material calibration is incomplete')
    return dict(passed=not reasons,failures=reasons,material_sha256=object_hash(mat),
         reference_damage_strains=strains,abaqus_formula='d1 + d2 * exp(-d3 * eta), eta=-p/q',
         scope='Necessary consistency screen, not experimental validation or proof over all triaxialities')
