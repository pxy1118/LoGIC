"""Explicit dimensional adaptation of the archived CFD recipe.

This creates a candidate protocol, not a production-validated solver setup.
"""
import math


def validate_cached_mesh_variant(original, candidate, reuse_mesh=False):
    """Permit a fresh mesh study without silently changing its cached geometry.

    Other physical experiments need a separate, explicitly supported route.
    This override is deliberately restricted to volume-mesh controls.
    """
    for section in ('source', 'geometry', 'physics', 'measurement', 'prediction'):
        if original[section] != candidate[section]:
            raise ValueError(f'Cached mesh variant changed {section}')
    if original.get('provenance', {}) != candidate.get('provenance', {}):
        raise ValueError('Cached mesh variant changed source provenance')
    if original['mesh']['surface_target_mm'] != candidate['mesh']['surface_target_mm']:
        raise ValueError('A cached surface cannot change its remesh target')
    if reuse_mesh and original['mesh'] != candidate['mesh']:
        raise ValueError('Changed mesh settings require a fresh volume mesh')
    sizes = [candidate['mesh'][name] for name in ('volume_hmin_mm', 'volume_hmax_mm')]
    if not all(isinstance(v, (int, float)) and math.isfinite(v) and v > 0 for v in sizes):
        raise ValueError('Volume mesh sizes must be finite and positive')
    if sizes[0] > sizes[1]:
        raise ValueError('Volume hmin exceeds hmax')


def legacy_aligned_candidate(specimen_size_mm=7.5):
    if not math.isfinite(specimen_size_mm) or specimen_size_mm <= 0:
        raise ValueError('Specimen size must be finite and positive')
    scale=specimen_size_mm/24.0
    inset=.1*scale
    buffer=.5*scale
    width=specimen_size_mm-2*inset
    length=specimen_size_mm+2*buffer
    return {
        'schema_version':1,
        'profile':'legacy_aligned_candidate',
        'status':'CANDIDATE_NOT_EXECUTED_NOT_FROZEN',
        'supersedes_production_candidate':'Stokes_0p1mm_s_fusion_exploration',
        'source':{'geometry_count':500,'split_counts':{'train':350,'validation':75,'test':75},
                  'paired_fea_count':433,'regenerate_solid':False,'mutate_fea_source':False},
        'geometry':{'specimen_size_mm':specimen_size_mm,'original_specimen_size_mm':24.0,
                    'dimensional_scale':scale,'side_inset_each_mm':inset,
                    'buffer_each_end_mm':buffer,'fluid_width_mm':width,
                    'flow_domain_length_mm':length,
                    'box_origin_mm':[inset,inset,-buffer],
                    'box_dimensions_mm':[width,width,length],
                    'boolean_backend':'Blender EXACT',
                    'surface_remesh_sequence':['adaptive','uniform','adaptive'],
                    'surface_triangle_target_mm':.15*scale,
                    'preserve_surface_contours':False,
                    'source_solid':'actual initial FEA solid surface',
                    'crop_note':'CFD side crop is inherited proportionally; FEA solid remains unchanged'},
        'physics':{'interface':'LaminarFlow','steady':True,'incompressible':True,
                   'include_inertia':True,'StokesFlowProp':False,
                   'density_kg_m3':1000.0,'viscosity_Pa_s':.001,
                   'inlet_velocity_m_s':.001,'outlet_pressure_Pa':0.0,
                   'wall':'NoSlip','outlet_normal_flow_constraint':False,
                   'outlet_suppress_backflow':True,'velocity_pressure_order':'P1/P1'},
        'mesh':{'surface_target_mm':.15*scale,'volume_hmax_mm':.8*scale,
                'volume_hmin_mm':.3*scale,'hgrad':1.15,'hcurve':.6,'hnarrow':.7,
                'policy':'same legacy controls and relative sizes; validate on new geometry'},
        'measurement':{'pressure_sections':'outer inlet and outlet buffer ends',
                       'reference_area_m2':(width*.001)**2,
                       'measurement_length_m':length*.001,
                       'nominal_specimen_length_m':specimen_size_mm*.001,
                       'permeability_scope':'apparent flow-domain permeability at the prescribed inlet speed',
                       'primary_K_field':'K_domain_app_z_m2',
                       'pressure_gradient_unit':'Pa/m',
                       'use_actual_signed_flux':True,
                       'formula':'K = mu*(Q/A)*L_meas/(p_in-p_out)',
                       'legacy_20mm_parameters_copied':False},
        'prediction':{'main_model':'existing Pore-GNN','voxel_size':120,
                      'voxel_representation':'binary initial-solid occupancy; sampling rule to freeze',
                      'targets':['ln(K_domain_app_z_m2 / 1 m^2)','pressure_gradient_Pa_m'],
                      'log_base':'e','inherit_config':'GNN/configs/default.yaml',
                      'architecture_search':False,'CGV_main_model':False,
                      'split_source':'existing geometry IDs and groups, not the old random 80/20 loader',
                      'pretraining_and_scalers':'training split only',
                      'checkpoint_selection':'validation only','test_policy':'once after method freeze'},
        'development':{'preserve_previous_runs':True,'reuse_Stokes_labels_for_production':False,
                       'recompute_after_protocol_change':True,'full_queue_started':False,
                       'checks':['surface deviation and pore connectivity','mass conservation',
                                 'representative mesh sensitivity','velocity dependence and Reynolds number'],
                       'K_mesh_sensitivity_target_rel':.02,'mass_imbalance_target_rel':.001},
    }
