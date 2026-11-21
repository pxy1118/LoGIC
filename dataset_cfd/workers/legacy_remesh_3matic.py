"""Archived adaptive/uniform/adaptive surface remeshing; no volume meshing."""
import os
import json
import time
import traceback
import sys
from pathlib import Path
import trimatic

case=Path(os.environ['CFD_V2_CASE']).resolve()
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root))
from src.surface_io import write_ascii_stl64
if not case.is_relative_to(root/'runs'):raise RuntimeError('Case outside CFD runs')
cfg=json.loads((case/'protocol.json').read_text(encoding='utf8'))['geometry']
working_scale=cfg.get('processing_scale',1.0)
r={'status':'STARTING','stages':[],'geometry_config':cfg};started=time.time()
try:
    trimatic.new_project()
    if cfg['boolean_backend']=='3-matic':
        solid=trimatic.import_part_stl(str(case/('solid_processing.stl' if working_scale!=1 else 'solid_initial.stl')),fix_normals=True)
        if isinstance(solid,(list,tuple)):solid=solid[0]
        box=trimatic.create_box_part(tuple(x*working_scale for x in cfg['box_origin_mm']),
                                    *[x*working_scale for x in cfg['box_dimensions_mm']])
        part=trimatic.boolean_subtraction(box,solid)
    else:
        part=trimatic.import_part_stl(str(case/('fluid_boolean_working.stl' if working_scale!=1 else 'fluid_boolean_raw.stl')),fix_normals=True)
    if isinstance(part,(list,tuple)):
        if len(part)!=1:raise RuntimeError('Multiple fluid parts require connectivity review')
        part=part[0]
    if cfg['boolean_backend']=='3-matic':
        part.name='fluid_boolean_working' if working_scale!=1 else 'fluid_boolean_raw'
        if working_scale!=1 or cfg.get('surface_export_format')!='ascii_float64':
            trimatic.export_stl_binary([part],str(case))
    def metrics():
        return {'volume_mm3':float(part.volume)/working_scale**3,'bad_edges':int(part.number_of_bad_edges),
                'self_intersections':int(trimatic.get_number_of_intersecting_triangles(part))}
    r['raw']=metrics()
    if cfg.get('surface_export_format')=='ascii_float64':
        vertices,triangles=part.get_triangles()
        r['raw_float64_export']=write_ascii_stl64(case/'fluid_boolean_raw.stl',vertices,triangles,working_scale)
    if r['raw']['bad_edges']:
        raise RuntimeError('Blender Boolean has open/nonmanifold edges')
    # The legacy remeshing stage may remove intersecting/sliver triangles.
    # Record intermediate defects; only the final inspected surface can be
    # accepted. Never pass an intersecting surface on to the CFD solver.
    for operation in cfg['surface_remesh_sequence']:
        t=time.time()
        print('STARTING_REMESH',operation,'target_mm',cfg['surface_triangle_target_mm'],
              'preserve_contours',cfg['preserve_surface_contours'],flush=True)
        if operation=='adaptive' and 'adaptive_max_geom_error_mm' in cfg:
            trimatic.adaptive_remesh_expert(entities=[part],
                maximum_geometrical_error=cfg['adaptive_max_geom_error_mm']*working_scale,
                minimum_triangle_edge_length=.0001*working_scale,
                maximum_triangle_edge_length=cfg['surface_triangle_target_mm']*working_scale,
                number_of_iterations=3,preserve_surface_contours=cfg['preserve_surface_contours'],
                preserve_sharp_edges=True)
        else:
            method=trimatic.adaptive_remesh if operation=='adaptive' else trimatic.uniform_remesh
            kwargs={'entities':[part],'target_triangle_edge_length':cfg['surface_triangle_target_mm']*working_scale,
                    'preserve_surface_contours':cfg['preserve_surface_contours']}
            if operation=='uniform' and 'uniform_sharp_edge_angle_deg' in cfg:
                kwargs['preserve_sharp_edge_angle']=cfg['uniform_sharp_edge_angle_deg']
            method(**kwargs)
        m=metrics();r['stages'].append({'operation':operation,'elapsed_s':time.time()-t,**m})
        print(r['stages'][-1],flush=True)
        part.name='surface_pass_%02d'%len(r['stages'])
        trimatic.export_stl_binary([part],str(case))
    m=metrics()
    r['final']=m
    if m['bad_edges'] or m['self_intersections']:
        raise RuntimeError('Final surface remains invalid after %d configured remesh stages'%len(r['stages']))
    part.name='fluid_working' if working_scale!=1 else 'fluid_domain'
    if working_scale!=1 or cfg.get('surface_export_format')!='ascii_float64':
        trimatic.export_stl_binary([part],str(case))
    if cfg.get('surface_export_format')=='ascii_float64':
        vertices,triangles=part.get_triangles()
        r['final_float64_export']=write_ascii_stl64(case/'fluid_domain.stl',vertices,triangles,working_scale)
    r.update(status='SURFACE_REMESH_EXPORTED' if cfg['surface_remesh_sequence'] else 'SURFACE_EXPORTED_WITHOUT_REMESH',final=metrics())
except Exception:
    r.update(status='FAILED',error=traceback.format_exc());traceback.print_exc()
finally:
    r['elapsed_s']=time.time()-started
    (case/'remesh_report.json').write_text(json.dumps(r,indent=2),encoding='utf8')
    print(r,flush=True)
    trimatic.exit()
