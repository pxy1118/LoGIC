"""Development mesh checks using an explicit, recorded quality definition."""
import math


def volume_mesh_report(stats, expected_volume_m3, specimen_length_m, surface_mode):
    types=list(map(str,stats.getTypes()))
    if 'tet' not in types:
        raise ValueError('CFD mesh contains no tetrahedra')
    # The API initially reports volcircum, whereas COMSOL's meshing log uses
    # skewness. Never compare those two values as if they were the same metric.
    initial_measure=str(stats.getQualityMeasure())
    stats.setQualityMeasure('volcircum')
    volcircum_min=float(stats.getMinQuality('tet'))
    stats.setQualityMeasure('skewness')
    volume=float(stats.getVolume('tet'))
    r={'elements':int(stats.getNumElem('tet')),
       'min_quality':float(stats.getMinQuality('tet')),
       'mean_tet_quality':float(stats.getMeanQuality('tet')),
       'min_tet_volume_m3':float(stats.getMinVolume('tet')),
       'quality_measure':str(stats.getQualityMeasure()),
       'initial_api_quality_measure':initial_measure,
       'min_tet_volcircum_quality':volcircum_min,
       'quality_scope':'tetrahedral volume elements only',
       'all_types_min_quality':float(stats.getMinQuality()),
       'volume_m3':volume,'expected_volume_m3':expected_volume_m3,
       'volume_closure_rel':abs(volume-expected_volume_m3)/specimen_length_m**3,
       'surface_remesh_mode':surface_mode,
       'element_counts_by_type':{t:int(stats.getNumElem(t)) for t in types}}
    if 'tri' in types:r['boundary_min_quality']=float(stats.getMinQuality('tri'))
    # Poor elements are retained as an explicit review requirement. Passing
    # these checks authorizes a sensitivity solve, never a production label.
    r['quality_review_required']=bool(r['min_quality']<.001 or volcircum_min<.001)
    r['production_accepted']=False
    r['accepted_for_development_solve']=bool(r['elements']>0 and
        all(math.isfinite(r[k]) and r[k]>0 for k in ['min_quality','min_tet_volume_m3','volume_m3']) and
        r['volume_closure_rel']<=.001)
    return r
