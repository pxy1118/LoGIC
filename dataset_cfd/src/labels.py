import math


def permeability(q_in, q_out, pressure_in, pressure_out, area, length, viscosity=0.001):
    vals=(q_in,q_out,pressure_in,pressure_out,area,length,viscosity)
    if not all(math.isfinite(v) for v in vals):
        raise ValueError('Non-finite flow measurements')
    dp=pressure_in-pressure_out
    if min(area,length,viscosity,q_out,dp)<=0 or q_in>=0:
        raise ValueError('Invalid dimensions or inconsistent signed flux/pressure direction')
    K=viscosity*q_out*length/(area*dp)
    imbalance=abs(q_in+q_out)/max(abs(q_in),abs(q_out))
    return {'K_app_z_m2':K,'log10_K_app_z':math.log10(K),
            'delta_p_sample_Pa':dp,'pressure_gradient_z_Pa_m':dp/length,
            'Q_in_m3_s':q_in,'Q_out_m3_s':q_out,'Q_m3_s':q_out,
            'U_superficial_m_s':q_out/area,'reference_area_m2':area,
            'sample_length_m':length,'viscosity_Pa_s':viscosity,
            'mass_imbalance_rel':imbalance,'cfd_valid':imbalance<=0.001,
            'logK_valid':imbalance<=0.001,'flow_solution_available':True}
