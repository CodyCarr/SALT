/* SPDX-License-Identifier: BSD-3-Clause */

#pragma once

/*
 * Internal interfaces shared by the line-profile driver and the numerical
 * absorption/emission kernels.  These functions are implementation details;
 * applications should include salt.h and call Line_Profile().
 */

void computeABS_vector(
    double wavelength, double oscillator_strength,
    double einstein_coefficient, const double *v_obs, int n,
    double alpha, double psi, double gamma, double tau,
    double v_0, double v_w, double v_ap, double v_b,
    double covering_fraction, double delta, int aperture,
    double sobolev_width, int use_sobolev_wings,
    int profile_method, double *out
);

void computeEM_vector(
    double wavelength, double emitted_wave,
    const double *blended_waves, const double *blended_osc_strs,
    const double *blended_abs_ein, const int *blended_fluor,
    const double *blended_p_r, const double *blended_p_f,
    const double *blended_fluor_waves, const int *n_fluor_each,
    int n_blends, double oscillator_strength, double einstein_coefficient,
    double lambda_ref, const double *v_obs, int n,
    const double *shell_luminosity, int n_shells,
    double alpha, double psi, double gamma, double tau,
    double v_0, double v_w, double v_ap, double v_b,
    double covering_fraction, double k_dust, double delta,
    int aperture, int resonance, int fluorescence, int blending,
    int occultation, int profile_method, double p_r, double p_f,
    double *out
);
