"""Executed by an isolated 3-matic 18 process; case path arrives via environment."""
import contextlib
import json
import os
from pathlib import Path
import time
import sys
import traceback
import trimatic


def single(value):
    if isinstance(value,(tuple,list)):
        if len(value)!=1: raise RuntimeError('Expected one part')
        return value[0]
    return value


def strict_overlaps(part,tolerance):
    trimatic.mark_entities(part,mark=False)
    trimatic.mark_overlapping_triangles(part,tolerance=tolerance)
    count=int(part.number_of_marked_triangles)
    trimatic.mark_entities(part,mark=False)
    return count


def process_case(case,session=None):
    case=Path(case).resolve()
    cfg=json.loads((case/'mesher_config.json').read_text(encoding='utf-8'))
    started=time.time()
    details={}
    with (case/'mesher.log').open('w',encoding='utf-8',buffering=1) as log:
        with contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
            try:
                trimatic.new_project()
                if trimatic.get_parts(): raise RuntimeError('Failed to clear 3-matic project')
                details['empty_project_before']=True
                if (case/'volume.inp').exists() or (case/'volume_3matic.inp').exists(): raise RuntimeError('Existing volume preserved')
                part=single(trimatic.import_part_stl(str(case/'source.stl'),fix_normals=True))
                details['source_intersections']=int(trimatic.get_number_of_intersecting_triangles(part))
                details['source_near_parallel_triangles_0p1mm']=int(trimatic.get_number_of_overlapping_triangles(part))
                details['source_overlaps']=strict_overlaps(part,cfg.get('overlap_tolerance_mm',1e-5))
                # Raw Boolean caps can contain needle triangles; remeshing must resolve them.
                # Acceptance applies to the exported remesh and volume, with volume/bounds checks.
                details['repairs']=[]
                wrap=cfg.get('surface_wrap',{})
                if wrap.get('enabled'):
                    volume=float(part.volume)
                    parameters=dict(gap_closing_distance=wrap['gap_closing_distance_mm'],
                        smallest_detail=wrap['smallest_detail_mm'],
                        protect_thin_walls=wrap.get('protect_thin_walls',False),
                        resulting_offset=0,reduce=wrap.get('reduce',True),
                        preserve_sharp_features=wrap.get('preserve_sharp_features',False),
                        preserve_surface_structure=False)
                    part=single(trimatic.wrap(entities=part,**parameters))
                    details['repairs'].append(dict(stage='before_remesh',method='Wrap',parameters=parameters,
                        volume_before_mm3=volume,volume_after_mm3=float(part.volume)))
                    part.name='wrapped';trimatic.export_stl_binary([part],str(case))
                    details['wrapped_surface_check']=dict(self_intersections=int(trimatic.get_number_of_intersecting_triangles(part)),
                        overlapping_triangles=strict_overlaps(part,cfg.get('overlap_tolerance_mm',1e-5)),
                        bad_edges=int(part.number_of_bad_edges),volume_mm3=float(part.volume))
                filter_distance=cfg.get('sharp_triangle_filter_mm',0)
                def filter_needles(part,stage):
                    volume=float(part.volume)
                    fixed=single(trimatic.filter_sharp_triangles(part,filter_distance=filter_distance,threshold_angle=5))
                    details['repairs'].append(dict(stage=stage,volume_before_mm3=volume,volume_after_mm3=float(fixed.volume),
                             method='collapse sharp triangles',filter_distance_mm=filter_distance))
                    return fixed
                if filter_distance>0:part=filter_needles(part,'before_remesh')
                def repair(part,stage):
                    volume=float(part.volume)
                    fixed=single(trimatic.auto_fix(part))
                    details['repairs'].append(dict(stage=stage,volume_before_mm3=volume,
                        volume_after_mm3=float(fixed.volume),method='3-matic auto_fix: filtering and small-shell removal'))
                    return fixed
                if cfg.get('repair_surface_defects') and (details['source_intersections'] or details['source_overlaps']):
                    part=repair(part,'before_remesh')
                if cfg.get('remesh_method')=='sequence':
                    steps=cfg.get('surface_steps')
                    if not isinstance(steps,list) or not steps:raise ValueError('A nonempty surface_steps list is required')
                    if wrap.get('enabled') or filter_distance or cfg.get('repair_surface_defects') or cfg.get('small_edge_filter_steps'):
                        raise ValueError('Sequence mode uses only its explicit no-Wrap steps')
                    sys.path.insert(0,str(Path(__file__).resolve().parent))
                    from surface_ops_3matic import apply
                    details['surface_stages']=[]
                    for index,step in enumerate(steps,1):
                        start=time.time();volume=float(part.volume)
                        part=apply(part,step)
                        part.name='surface_stage%02d'%index
                        trimatic.export_stl_binary([part],str(case))
                        filename=part.name+'.stl'
                        # Match the surface trials: each next stage consumes its serialized predecessor.
                        trimatic.delete(part)
                        part=single(trimatic.import_part_stl(str(case/filename),fix_normals=True))
                        details['surface_stages'].append(dict(step=step,file=filename,
                            volume_before_mm3=volume,volume_after_mm3=float(part.volume),
                            self_intersections=int(trimatic.get_number_of_intersecting_triangles(part)),
                            overlapping_triangles=strict_overlaps(part,cfg.get('overlap_tolerance_mm',1e-5)),
                            bad_edges=int(part.number_of_bad_edges),elapsed_s=time.time()-start))
                elif cfg.get('remesh_method')=='uniform':
                    part=single(trimatic.uniform_remesh(entities=part,
                        target_triangle_edge_length=cfg['surface_edge_mm'],
                        preserve_surface_contours=cfg.get('surface_preserve_contours',False),
                        preserve_sharp_edge_angle=cfg.get('surface_sharp_edge_angle_deg',45),
                        split_edge_factor=cfg.get('surface_split_edge_factor',.4),skip_bad_edges=False))
                elif cfg.get('remesh_method')=='expert':
                    part=single(trimatic.adaptive_remesh_expert(entities=part,
                        maximum_triangle_edge_length=cfg['surface_edge_mm'],
                        minimum_triangle_edge_length=cfg.get('surface_minimum_edge_mm',0),
                        maximum_geometrical_error=cfg['maximum_geometrical_error_mm'],
                        number_of_iterations=3,preserve_surface_contours=True,preserve_sharp_edges=True))
                elif cfg.get('remesh_method','adaptive')=='adaptive':
                    part=single(trimatic.adaptive_remesh(entities=part,
                        target_triangle_edge_length=cfg['surface_edge_mm'],
                        preserve_surface_contours=cfg.get('surface_preserve_contours',True)))
                else:
                    raise ValueError('Unknown remesh method: '+str(cfg.get('remesh_method')))
                if filter_distance>0:part=filter_needles(part,'after_remesh')
                for index, step in enumerate(cfg.get('small_edge_filter_steps', []), 1):
                    volume=float(part.volume)
                    part=single(trimatic.filter_small_edges(entities=part,
                        filter_distance=step['filter_distance_mm'],threshold_angle=30,
                        number_of_iterations=3,surface_border_only=False))
                    details['repairs'].append(dict(stage='after_remesh_small_edges_%d' % index,
                        method='collapse small edges',filter_distance_mm=step['filter_distance_mm'],
                        threshold_angle=30,number_of_iterations=3,surface_border_only=False,
                        volume_before_mm3=volume,volume_after_mm3=float(part.volume)))
                intersections=int(trimatic.get_number_of_intersecting_triangles(part))
                details['remesh_near_parallel_triangles_0p1mm']=int(trimatic.get_number_of_overlapping_triangles(part))
                overlaps=strict_overlaps(part,cfg.get('overlap_tolerance_mm',1e-5))
                details.update(remesh_intersections_before_repair=intersections,remesh_overlaps_before_repair=overlaps)
                if cfg.get('repair_surface_defects') and (intersections or overlaps):
                    part=repair(part,'after_remesh')
                    intersections=int(trimatic.get_number_of_intersecting_triangles(part))
                    overlaps=strict_overlaps(part,cfg.get('overlap_tolerance_mm',1e-5))
                part.name='remeshed'
                trimatic.export_stl_binary([part],str(case))
                details.update(self_intersections=intersections,overlapping_triangles=overlaps)
                if intersections or overlaps:
                    raise RuntimeError('Remeshed surface rejected: %d intersecting and %d overlapping triangles' % (intersections, overlaps))
                # Check the file consumed by downstream QA, including STL round-off.
                exported=single(trimatic.import_part_stl(str(case/'remeshed.stl'),fix_normals=True))
                exported_intersections=int(trimatic.get_number_of_intersecting_triangles(exported))
                exported_overlaps=strict_overlaps(exported,cfg.get('overlap_tolerance_mm',1e-5))
                details['exported_surface_check']=dict(self_intersections=exported_intersections,
                    overlapping_triangles=exported_overlaps,volume_mm3=float(exported.volume),
                    bad_edges=int(exported.number_of_bad_edges))
                exported_bad_edges=int(exported.number_of_bad_edges)
                trimatic.delete(exported)
                if exported_intersections or exported_overlaps or exported_bad_edges:
                    raise RuntimeError('Exported STL rejected: %d intersecting, %d overlapping triangles, %d bad edges' % (exported_intersections, exported_overlaps, exported_bad_edges))
                part.name='SPECIMEN'
                trimatic.create_volume_mesh(part=part,element_type=trimatic.ElementType.Tet10,
                                           project_midpoint_entity=None,maximum_edge_length=cfg['volume_edge_mm'])
                filename='volume_3matic.inp' if cfg.get('midside_policy','native')=='exact_edge_midpoints' else 'volume.inp'
                trimatic.export_abaqus_single_output([part],str(case/filename),False,
                    trimatic.TypeOfElement.element_type_C3D10,None,trimatic.FaceSplitMethod.split_by_part)
                result=dict(status='MESH_EXPORTED',self_intersections=intersections,
                            midpoint_projection='disabled',
                            overlapping_triangles=overlaps,overlap_tolerance_mm=cfg.get('overlap_tolerance_mm',1e-5),volume_mm3=float(part.volume))
            except Exception:
                result=dict(status='FAILED',error=traceback.format_exc()); traceback.print_exc()
            finally:
                if session is not None:
                    try:
                        trimatic.new_project()
                        if trimatic.get_parts(): raise RuntimeError('Parts remain after clearing project')
                        details['empty_project_after']=True
                    except Exception:
                        result.update(status='FAILED',error=traceback.format_exc(),cleanup_failed=True)
                    result['session']=session
                result['elapsed_s']=time.time()-started
                result.update(details)
                temp=case/'mesher_result.json.tmp'; temp.write_text(json.dumps(result,indent=2),encoding='utf-8')
                temp.replace(case/'mesher_result.json')
    
    return result


if __name__=='__main__':
    try:process_case(os.environ['FEA_V2_CASE'])
    finally:trimatic.exit()
