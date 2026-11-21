"""Verify saved model settings and successful solution against the pilot profile."""
from pathlib import Path
import argparse
import sys
import zipfile
import re
import xml.etree.ElementTree as ET
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.common import ROOT,read,save,now

p=argparse.ArgumentParser();p.add_argument('--case',type=Path,required=True);a=p.parse_args()
case=a.case.resolve()
if not case.is_relative_to(ROOT/'runs'):raise ValueError('Case outside CFD runs')
cfg=read(case/'protocol.json');result=read(case/'results.json')
with zipfile.ZipFile(case/'fluid_presolution.mph') as archive:
    root=ET.fromstring(archive.read('dmodel.xml'))
flow=next(n for n in root.iter('Physics') if n.attrib.get('tag')=='spf')
def scalar(value):
    tokens=re.findall("'([^']*)'",value or '')
    return tokens[0] if tokens else value
parameters={n.attrib['param']:scalar(n.attrib.get('value')) for n in flow.iter('param')}
globals={n.attrib['name']:n.attrib['expr'] for n in root.iter('expressions') if 'expr' in n.attrib}
features={n.attrib['tag']:{x.attrib['param']:scalar(x.attrib.get('value')) for x in n.iter('param')}
          for n in flow.iter('PhysicsFeature')}
checks={'navier_stokes':flow.attrib['op']=='LaminarFlow' and parameters.get('StokesFlowProp')=='0',
        'incompressible':parameters.get('Compressibility')=='Incompressible',
        'gravity_disabled':parameters.get('IncludeGravity')=='0',
        'inlet_1mm_s':globals.get('U0')=='0.001[m/s]' and features['inl1'].get('U0in')=='U0',
        'density':globals.get('rho_f')=='1000[kg/m^3]',
        'viscosity':globals.get('mu_f')=='1e-3[Pa*s]',
        'p1_p1':parameters.get('order_fluid')=='1',
        'outlet_zero_pressure':features['out1'].get('p0') in ['0','0[Pa]'],
        'actual_solution':max(result['solution_size'])>1,
        'mass_balance':result['mass_imbalance_rel']<=.001,
        'independent_K_check':result['template_kappa_relative_difference']<=.001}
save(case/'physics_verification.json',{'utc':now(),'checks':checks,'passed':all(checks.values()),
     'nominal_inlet_velocity_m_s':cfg['physics']['inlet_velocity_m_s'],
     'actual_superficial_velocity_m_s':result['U_superficial_m_s'],
     'production_protocol_frozen':False})
print(checks)
if not all(checks.values()):raise RuntimeError('Saved physical configuration verification failed')
