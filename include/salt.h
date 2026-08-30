/* SPDX-License-Identifier: BSD-3-Clause */

#pragma once

#if defined(_WIN32)
#  define SALT_API __declspec(dllexport)
#elif defined(__GNUC__) || defined(__clang__)
#  define SALT_API __attribute__((visibility("default")))
#else
#  define SALT_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Public option values accepted by Line_Profile(). */
enum salt_profile_method { SALT_VOIGT_WOFZ = 0, SALT_VOIGT_COLT = 1 };
enum salt_profile_type {
    SALT_PROFILE_ABSORPTION = 0,
    SALT_PROFILE_EMISSION = 1,
    SALT_PROFILE_PCYGNI = 2
};

/*
 * Compute a continuum-normalized SALT line profile.
 *
 * Units:
 *   - wavelengths: Angstrom
 *   - v_obs, v_0, v_w, v_ap, v_b and sobolev_width: km s^-1
 *   - alpha and psi: radians
 *
 * Arrays are borrowed for the duration of the call and must contain at least
 * the lengths supplied alongside them.  out_profile must hold nV doubles.
 * The Python wrapper is the recommended interface for ordinary use.
 * If use_sobolev_wings is zero, v_b must be positive; otherwise no finite
 * Voigt width exists and the absorption kernel returns zero.
 */
SALT_API void Line_Profile(
    const double *v_obs, int nV,
    double lambda_ref,
    const double *background,

    const double *abs_waves, const double *abs_osc, const double *abs_ein, int nAbs,
    const double *em_waves,
    const double *emitted_waves,
    const double *em_osc,
    const double *em_ein,
    int nEm,

    const int *res,
    const int *fluor,
    const int *blending,
    const double *p_r_arr,
    const double *p_f_arr,

    const double *blended_waves,
    const double *blended_osc_strs,
    const double *blended_abs_ein,
    const int *blended_fluor,
    const double *blended_p_r,
    const double *blended_p_f,
    const double *blended_fluor_waves,
    const int *n_fluor_each,
    const int *n_blends_each,

    const int *line_num, int nLineNum,

    double alpha, double psi, double gamma, double tau,
    double v_0, double v_w, double v_ap, double v_b,
    double covering_fraction, double k_dust, double delta,
    int aperture, int occultation,
    double sobolev_width,
    int use_sobolev_wings,
    int profile_method,

    int profile_type,
    double *out_profile
);

#ifdef __cplusplus
}
#endif
