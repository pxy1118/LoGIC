"""Explicit surface operations for frozen, no-Wrap remeshing sequences."""
import trimatic


def one(value,previous):
    if value is None:return previous
    if isinstance(value,(tuple,list)):
        if len(value)!=1:raise RuntimeError('Expected exactly one surface part')
        return value[0]
    return value


def apply(part,step):
    operation=step['operation']
    if operation=='uniform':
        value=trimatic.uniform_remesh(entities=part,target_triangle_edge_length=step['edge_mm'],
            preserve_surface_contours=step.get('preserve_contours',False),
            preserve_sharp_edge_angle=step.get('sharp_angle_deg',45),
            split_edge_factor=step.get('split_factor',.4),skip_bad_edges=False)
    elif operation=='adaptive':
        value=trimatic.adaptive_remesh(entities=part,target_triangle_edge_length=step['edge_mm'],
            preserve_surface_contours=step.get('preserve_contours',False))
    elif operation=='expert':
        value=trimatic.adaptive_remesh_expert(entities=part,
            maximum_triangle_edge_length=step['edge_mm'],minimum_triangle_edge_length=step.get('minimum_mm',0),
            maximum_geometrical_error=step.get('error_mm',.005),
            shape_measure=getattr(trimatic.ShapeMeasures,step.get('quality_measure','skewness_n')),
            shape_quality_threshold=step.get('quality_threshold',.3),
            number_of_iterations=step.get('iterations',3),skip_bad_edges=False,
            preserve_surface_contours=step.get('preserve_contours',True),
            preserve_sharp_edges=step.get('preserve_sharp',True))
    elif operation=='small_edges':
        value=trimatic.filter_small_edges(entities=part,filter_distance=step['distance_mm'],
            threshold_angle=step.get('threshold_angle_deg',30),number_of_iterations=step.get('iterations',3),
            surface_border_only=step.get('border_only',False))
    elif operation=='sharp_triangles':
        value=trimatic.filter_sharp_triangles(entities=part,filter_distance=step['distance_mm'],
            threshold_angle=step.get('threshold_angle_deg',5),filter_mode=trimatic.FilterSharpTriangleMode.Collapse)
    elif operation=='quality_reduce':
        value=trimatic.quality_preserving_reduce_triangles(entities=part,
            maximum_geometrical_error=step.get('error_mm',.005),maximum_triangle_edge_length=step['edge_mm'],
            shape_measure=getattr(trimatic.ShapeMeasures,step.get('quality_measure','skewness_n')),
            shape_quality_threshold=step.get('quality_threshold',.5),number_of_iterations=step.get('iterations',3),
            preserve_surface_contours=step.get('preserve_contours',False),skip_bad_edges=False)
    else:raise ValueError('Unsupported no-Wrap surface operation: '+operation)
    return one(value,part)
