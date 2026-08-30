/* SPDX-License-Identifier: BSD-3-Clause */

/*
 * Assemble a complete SALT spectrum from the absorption and scattered-
 * emission kernels.
 *
 * The driver applies wavelength offsets for multiplets, accumulates the
 * continuum removed in successive radial shells, distributes that luminosity
 * among resonant/fluorescent channels, and optionally convolves the emitted
 * spectrum with the one-dimensional Maxwellian velocity distribution.  The
 * physical model is described by Carr et al. (2023, ApJ, 952, 88) and the
 * thermal/microturbulent extension documented with this release.
 */

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "salt.h"
#include "salt_grid_config.h"
#include "salt_internal.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#ifndef M_PI
#define M_PI 3.141592653589793238462643383279502884
#endif

#ifndef MAX_BLENDS
#define MAX_BLENDS 32
#endif
#ifndef MAX_FLUOR
#define MAX_FLUOR 16
#endif

/* Fixed quadrature and sampling resolutions */
enum {
    Ns = SALT_LINE_NS,  /* Shell radii used to compute shell energies. */
    Nyo = SALT_LINE_NYO,  /* Samples used for the covering-fraction floor. */
    Ny_offset = SALT_LINE_FIRST_SHELL_OFFSET
};

/* ============================================================
   Helpers matching NumPy/SciPy behavior
   ============================================================ */

static inline int searchsorted_left(const double *a, int n, double x)
{
    int lo = 0, hi = n;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (a[mid] < x) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}


/*------------------------------------------------------------
  Shift a spectrum by an arbitrary velocity using linear interpolation.
  Values shifted outside the grid are set to zero.
------------------------------------------------------------*/

static inline void shift_profile(
    double *dst,
    const double *src,
    const double *v_obs,
    int n,
    double vel_shift
)
{
    if (n < 2) {
        if (n == 1) dst[0] = 0.0;
        return;
    }

    const double dx = (v_obs[n-1] - v_obs[0]) / (double)(n-1);
    int uniform = dx > 0.0;
    if (uniform) {
        const double tol = 32.0 * 2.2204460492503131e-16
                         * fmax(1.0, fabs(v_obs[n-1]));
        for (int i = 1; i < n-1; i++) {
            if (fabs(v_obs[i] - (v_obs[0] + dx*(double)i)) > tol) {
                uniform = 0;
                break;
            }
        }
    }

    if (uniform) {
        const double inv_dx = 1.0 / dx;
        #ifdef _OPENMP
        #pragma omp parallel for schedule(static)
        #endif
        for (int i = 0; i < n; i++) {
            const double x = v_obs[i] - vel_shift;
            if (x <= v_obs[0] || x >= v_obs[n-1]) {
                dst[i] = 0.0;
                continue;
            }

            double u = (x - v_obs[0]) * inv_dx;
            int lo = (int)floor(u);
            if (lo < 0) lo = 0;
            if (lo > n-2) lo = n-2;
            u -= (double)lo;
            dst[i] = src[lo] + u*(src[lo+1] - src[lo]);
        }
        return;
    }

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < n; i++) {

        double x = v_obs[i] - vel_shift;

        if (x <= v_obs[0] || x >= v_obs[n-1]) {
            dst[i] = 0.0;
            continue;
        }

        int hi = searchsorted_left(v_obs,n,x);

        int lo = hi-1;

        double t = (x-v_obs[lo])/(v_obs[hi]-v_obs[lo]);

        dst[i] = src[lo] + t*(src[hi]-src[lo]);
    }
}

/* trapezoid(1 - flux, x=v_obs) */
static inline double trapz_one_minus(const double *flux, const double *x, int n)
{
    double area = 0.0;
    for (int i = 1; i < n; i++) {
        const double y0 = 1.0 - flux[i-1];
        const double y1 = 1.0 - flux[i];
        area += 0.5 * (y0 + y1) * (x[i] - x[i-1]);
    }
    return area;
}

/* ============================================================
   Convolution
   ============================================================ */

static void convolve_same_gaussian_dx(
    double *signal,
    const double *v_obs,
    int n,
    double sigma
)
{
    if (sigma<=0.0 || n<2)
        return;

    const double dx = fabs(v_obs[1]-v_obs[0]);

    if (dx<=0.0)
        return;

    /* ±5 sigma contains essentially all power */

    const int half =
        (int)ceil(5.0*sigma/dx);

    const int nkernel =
        2*half+1;

    double *kernel =
        malloc((size_t)nkernel*sizeof(double));

    double *out =
        calloc((size_t)n,sizeof(double));

    if(!kernel || !out){
        free(kernel);
        free(out);
        return;
    }

    double sum=0.0;

    for(int k=-half;k<=half;k++){

        double dv=k*dx;

        double w=
            exp(-0.5*dv*dv/(sigma*sigma));

        kernel[k+half]=w;

        sum+=w;
    }

    for(int k=0;k<nkernel;k++)
        kernel[k]/=sum;

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for(int i=0;i<n;i++){

        double accum=0.0;

        for(int k=-half;k<=half;k++){

            int j=i-k;

            if((unsigned)j<(unsigned)n)
                accum+=signal[j]*kernel[k+half];
        }

        out[i]=accum;
    }

    memcpy(signal,out,(size_t)n*sizeof(double));

    free(kernel);
    free(out);
}

/* ============================================================
   Covering-fraction geometry for floor
   floor = min(1 - Co) = 1 - max(Co)
   where Co = min(1, f_top + f_bottom)
   and y_range = linspace(1, v_w/v_0, 100)
   ============================================================ */

static inline double clampd(double x, double lo, double hi)
{
    return (x < lo) ? lo : (x > hi) ? hi : x;
}

/* scalar translation of get_Area_single() */
static inline double get_Area_single_scalar(double y, double alpha, double psi)
{
    if (y <= 0.0) return 0.0;

    const double sinA = sin(alpha);
    const double sinP = sin(psi);
    const double cosP = cos(psi);

    const double theta_c = asin(clampd(1.0 / y, -1.0, 1.0));

    /* regimes */
    if (psi >= theta_c + alpha) return 0.0;
    if (psi + alpha <= theta_c) {
        return clampd(y*y*sinA*sinA, 0.0, 1.0);
    }
    if (alpha >= theta_c + psi) return 1.0;

    /* partial */
    if (fabs(sinP) < 1e-14) {
        /* degenerate; safe clamp */
        return clampd(y*y*sinA*sinA, 0.0, 1.0);
    }

    const double half_term = 0.5*(theta_c - psi + alpha);
    const double dd = 2.0*y*sin(half_term);

    const double angle_one = 0.5*(psi + alpha - theta_c);
    const double angle_two = 0.5*(psi + theta_c - alpha);

    const double kk = (dd*sin(angle_one)) / sinP;
    const double zz = (sin(angle_two)*dd*cosP) / sinP;

    const double LL = 2.0*y*sinA;
    const double DD = LL*cosP;

    const double aa = 0.5*DD;
    const double bb = 0.5*LL;

        if (fabs(aa) < 1e-14) return 0.0;

    const double t1 = (zz - aa) / aa;
    const double sqrt1 = sqrt(fmax(0.0, 1.0 - t1*t1));

    const double area_1 =
        bb * ( (zz-aa)*sqrt1 +
               aa*asin(clampd(t1, -1.0, 1.0)) +
               0.5*M_PI*aa );

    const double t2 = 1.0 - kk;
    const double sqrt2 = sqrt(fmax(0.0, -kk*kk + 2.0*kk));

    const double area_2 =
        (kk-1.0)*sqrt2 -
        asin(clampd(t2, -1.0, 1.0)) +
        0.5*M_PI;

    return clampd((area_1 + area_2) / M_PI, 0.0, 1.0);
}

static inline double get_Area_bicone_scalar(double y, double alpha, double psi)
{
    const double f_top = get_Area_single_scalar(y, alpha, psi);
    const double f_bot = get_Area_single_scalar(y, alpha, M_PI - psi);
    const double sum   = f_top + f_bot;
    return (sum > 1.0) ? 1.0 : sum;
}

static double covering_floor_scalar(double alpha, double psi,
                                    double v_0, double v_w)
{
    if (v_0 == 0.0) return 0.0;

    /* ---- Geometry must use magnitudes (signs are RT only) ---- */
    const double y_max = fabs(v_w) / fabs(v_0);

    /* ---- Clamp psi safely into [0, π/2) to avoid singularity ---- */
    const double HALF_PI = 0.5 * M_PI;
    const double EPS_PSI = 1e-12;

    if (psi < 0.0) psi = 0.0;
    if (psi >= HALF_PI) psi = HALF_PI - EPS_PSI;

    /* ---- Degenerate or trivial case ---- */
    if (Nyo < 2 || y_max <= 1.0) {
        const double y = 1.0 + 1e-12;
        const double Co = get_Area_bicone_scalar(y, alpha, psi);
        return clampd(1.0 - Co, 0.0, 1.0);
    }

    const double dy = (y_max - 1.0) / (double)(Nyo - 1);

    double Co_max = 0.0;

    /* ---- Skip y = 1 exactly (start at i=1) ---- */
    for (int i = 1; i < Nyo; i++) {

        const double y = 1.0 + dy * (double)i;

        const double Co = get_Area_bicone_scalar(y, alpha, psi);

        if (Co > Co_max)
            Co_max = Co;

        /* Early exit — cannot exceed 1 */
        if (Co_max >= 1.0)
            break;
    }

    return clampd(1.0 - Co_max, 0.0, 1.0);
}

/* ============================================================
   Absorption update core (matches makeAbsorptionPROFILE)
   - uses SAME alpha/psi as absorption run
   - floor computed once per run
   ============================================================ */

static void absorption_run_final_flux(
    const double *v_obs, int nV,
    double lambda_ref,
    const double *background,
    const double *abs_waves,
    const double *abs_osc,
    const double *abs_ein,
    int nAbs,
    double alpha_use, double psi_use,
    double gamma, double tau,
    double v0_use, double vw_use,
    double v_ap_use, double v_b,
    double f_holes, double delta,
    int APERTURE,
    double SW,
    int use_sobolev_wings,
    int profile_method,
    double *flux_out
)
{
    const double c_kms = 2.99792458e5;

    memcpy(flux_out, background, (size_t)nV * sizeof(double));

    const double floorv = covering_floor_scalar(alpha_use, psi_use, v0_use, vw_use);

    double *Iabs  = (double*)malloc((size_t)nV * sizeof(double));
    double *Iroll = (double*)malloc((size_t)nV * sizeof(double));
    if (!Iabs || !Iroll) { free(Iabs); free(Iroll); return; }

    for (int l = 0; l < nAbs; l++) {

        computeABS_vector(
            abs_waves[l], abs_osc[l], abs_ein[l],
            v_obs, nV,
            alpha_use, psi_use, gamma, tau,
            v0_use, vw_use,
            v_ap_use, v_b,
            f_holes, delta,
            APERTURE, SW, use_sobolev_wings, profile_method,
            Iabs
        );

	const double vel_shift =
	  c_kms*(abs_waves[l]-lambda_ref)/lambda_ref;

	shift_profile(
	    Iroll,
	    Iabs,
	    v_obs,
	    nV,
	    vel_shift
	);

        #ifdef _OPENMP
        #pragma omp parallel for schedule(static)
        #endif
        for (int i = 0; i < nV; i++) {
            double f = flux_out[i] + flux_out[i] * Iroll[i];
            if (f < floorv) f = floorv;
            flux_out[i] = f;
        }
    }

    free(Iabs);
    free(Iroll);
}

/* Same as above, but records the trapezoid areas after each line:
   areas[0] = trapz(1 - background)
   areas[l+1] = trapz(1 - flux_after_line_l)
*/
static void absorption_run_areas(
    const double *v_obs, int nV,
    double lambda_ref,
    const double *background,
    const double *abs_waves,
    const double *abs_osc,
    const double *abs_ein,
    int nAbs,
    double alpha_use, double psi_use,
    double gamma, double tau,
    double v0_use, double vw_use,
    double v_ap_use, double v_b,
    double f_holes, double delta,
    int APERTURE,
    double SW,
    int use_sobolev_wings,
    int profile_method,
    double *areas_out,          /* length nAbs+1 */
    double *final_flux_optional /* may be NULL */
)
{
    const double c_kms = 2.99792458e5;

    double *flux = (double*)malloc((size_t)nV * sizeof(double));
    double *Iabs  = (double*)malloc((size_t)nV * sizeof(double));
    double *Iroll = (double*)malloc((size_t)nV * sizeof(double));
    if (!flux || !Iabs || !Iroll) {
        free(flux); free(Iabs); free(Iroll);
        return;
    }

    memcpy(flux, background, (size_t)nV * sizeof(double));

    const double floorv = covering_floor_scalar(alpha_use, psi_use, v0_use, vw_use);

    areas_out[0] = trapz_one_minus(flux, v_obs, nV);

    for (int l = 0; l < nAbs; l++) {

        computeABS_vector(
            abs_waves[l], abs_osc[l], abs_ein[l],
            v_obs, nV,
            alpha_use, psi_use, gamma, tau,
            v0_use, vw_use,
            v_ap_use, v_b,
            f_holes, delta,
            APERTURE, SW, use_sobolev_wings, profile_method,
            Iabs
        );

	const double vel_shift =
	  c_kms*(abs_waves[l]-lambda_ref)/lambda_ref;

	shift_profile(
	    Iroll,
	    Iabs,
	    v_obs,
	    nV,
	    vel_shift
	);

        #ifdef _OPENMP
        #pragma omp parallel for schedule(static)
        #endif
        for (int i = 0; i < nV; i++) {
            double f = flux[i] + flux[i] * Iroll[i];
            if (f < floorv) f = floorv;
            flux[i] = f;
        }

        areas_out[l+1] = trapz_one_minus(flux, v_obs, nV);
    }

    if (final_flux_optional) {
        memcpy(final_flux_optional, flux, (size_t)nV * sizeof(double));
    }

    free(flux);
    free(Iabs);
    free(Iroll);
}

/* ============================================================
   Public API: Line_Profile
   ============================================================ */

void Line_Profile(
    const double *v_obs, int nV,
    double lambda_ref,
    const double *background,

    const double *abs_waves,
    const double *abs_osc,
    const double *abs_ein,
    int nAbs,

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
    const int    *blended_fluor,
    const double *blended_p_r,
    const double *blended_p_f,
    const double *blended_fluor_waves,
    const int    *n_fluor_each,
    const int    *n_blends_each,

    const int *line_num,
    int nLineNum,

    double alpha, double psi, double gamma, double tau,
    double v_0, double v_w, double v_ap, double v_b,
    double f_holes, double k_dust, double delta,
    int APERTURE, int OCCULTATION,
    double SW,
    int use_sobolev_wings,
    int profile_method,

    int profile_type,
    double *out_profile
)
{
    for (int i = 0; i < nV; i++) out_profile[i] = 0.0;

    if (nV <= 0 || nAbs < 0 || nEm < 0) return;
    if (nLineNum != nAbs) return;

    if (profile_type == 0) {
        absorption_run_final_flux(
            v_obs, nV, lambda_ref, background,
            abs_waves, abs_osc, abs_ein, nAbs,
            alpha, psi, gamma, tau,
            -v_0, -v_w, -v_ap, v_b,
            f_holes, delta, APERTURE, SW, use_sobolev_wings, profile_method,
            out_profile
        );
        return;
    }

    const size_t nAbs_z = (size_t)nAbs;
    const size_t nEm_z  = (size_t)nEm;
    const size_t Ns_z   = (size_t)Ns;
    const size_t nCol_z = (size_t)(nAbs + 1);

    const double alpha_shell = M_PI / 2.0;
    const double psi_shell = 0.0;
    const double y_inf = fabs(v_w / v_0);

    double shells[Ns];
    for (int s = 0; s < Ns; s++)
        shells[s] = 1.0 + (y_inf - 1.0) * (double)s / (double)(Ns - 1);

    shells[0] += y_inf / (double)Ny_offset;

    double *CDF = malloc(Ns_z * nCol_z * sizeof(double));
    if (!CDF) {
        free(CDF);
        return;
    }

    for (int s = 0; s < Ns; s++) {
        double *areas_row = &CDF[(size_t)s * nCol_z];
        absorption_run_areas(
            v_obs, nV, lambda_ref, background,
            abs_waves, abs_osc, abs_ein, nAbs,
            alpha_shell, psi_shell, gamma, tau,
            -v_0, -v_0 * shells[s], -v_ap, v_b,
            f_holes, delta, APERTURE, SW, use_sobolev_wings, profile_method,
            areas_row,
            NULL
        );

    }

    double *CDF2 = malloc(Ns_z * nAbs_z * sizeof(double));
    if (!CDF2) {
        free(CDF);
        return;
    }

    for (int s = 0; s < Ns; s++) {
        for (int l = 0; l < nAbs; l++) {
            CDF2[(size_t)s * nAbs_z + (size_t)l] =
                CDF[(size_t)s * nCol_z + (size_t)(l + 1)]
              - CDF[(size_t)s * nCol_z + (size_t)l];
        }
    }

    int sum_rep = 0;
    for (int i = 0; i < nAbs; i++) sum_rep += line_num[i];

    if (sum_rep != nEm) {
        free(CDF);
        free(CDF2);
        return;
    }

    double *shell_energies = malloc(nEm_z * Ns_z * sizeof(double));
    if (!shell_energies) {
        free(CDF);
        free(CDF2);
        return;
    }

    int eidx = 0;
    for (int i = 0; i < nAbs; i++) {
        for (int r = 0; r < line_num[i]; r++) {
            for (int s = 0; s < Ns; s++) {
                shell_energies[(size_t)eidx * Ns_z + (size_t)s] =
                    CDF2[(size_t)s * nAbs_z + (size_t)i];
            }
            eidx++;
        }
    }

    free(CDF);
    free(CDF2);

    double *Eline = malloc((size_t)nV * sizeof(double));
    double *Eroll = malloc((size_t)nV * sizeof(double));
    double *Emission = calloc((size_t)nV, sizeof(double));

    if (!Eline || !Eroll || !Emission) {
        free(shell_energies);
        free(Eline);
        free(Eroll);
        free(Emission);
        return;
    }

    const double c_kms = 2.99792458e5;

    for (int l = 0; l < nEm; l++) {
        const int b0 = l * MAX_BLENDS;
        const int bf0 = l * MAX_BLENDS * MAX_FLUOR;

        computeEM_vector(
            em_waves[l],
            emitted_waves[l],
            blended_waves       ? &blended_waves[b0]        : NULL,
            blended_osc_strs    ? &blended_osc_strs[b0]     : NULL,
            blended_abs_ein     ? &blended_abs_ein[b0]      : NULL,
            blended_fluor       ? &blended_fluor[b0]        : NULL,
            blended_p_r         ? &blended_p_r[b0]          : NULL,
            blended_p_f         ? &blended_p_f[bf0]         : NULL,
            blended_fluor_waves ? &blended_fluor_waves[bf0] : NULL,
            n_fluor_each        ? &n_fluor_each[b0]         : NULL,
            n_blends_each       ? n_blends_each[l]          : 0,
            em_osc[l],
            em_ein[l],
            lambda_ref,
            v_obs,
            nV,
            &shell_energies[(size_t)l * Ns_z],
            Ns,
            alpha, psi, gamma, tau,
            -v_0, -v_w, -v_ap, v_b,
            f_holes, k_dust, delta,
            APERTURE,
            res ? res[l] : 0,
            fluor ? fluor[l] : 0,
            blending ? blending[l] : 0,
            OCCULTATION,
            profile_method,
            p_r_arr ? p_r_arr[l] : 0.0,
            p_f_arr ? p_f_arr[l] : 0.0,
            Eline
        );

	const double vel_shift =
	  c_kms*(emitted_waves[l]-lambda_ref)/lambda_ref;

	shift_profile(
	    Eroll,
	    Eline,
	    v_obs,
	    nV,
	    vel_shift
	);

        for (int i = 0; i < nV; i++)
            Emission[i] += Eroll[i];

    }

    if ((profile_type == 1 && v_b > 0.0) ||
        (profile_type == 2 && nV > 1 && v_b > fabs(v_obs[1] - v_obs[0]))) {
        convolve_same_gaussian_dx(Emission, v_obs, nV, v_b/sqrt(2.0));
    }

    if (profile_type == 1) {
        memcpy(out_profile, Emission, (size_t)nV * sizeof(double));
    } else if (profile_type == 2) {
        double *Abs_user = malloc((size_t)nV * sizeof(double));

        if (!Abs_user) {
            memcpy(out_profile, Emission, (size_t)nV * sizeof(double));
        } else {
            absorption_run_final_flux(
                v_obs, nV, lambda_ref, background,
                abs_waves, abs_osc, abs_ein, nAbs,
                alpha, psi, gamma, tau,
                -v_0, -v_w, -v_ap, v_b,
                f_holes, delta, APERTURE, SW, use_sobolev_wings, profile_method,
                Abs_user
            );

            for (int i = 0; i < nV; i++)
                out_profile[i] = Abs_user[i] + Emission[i];

            free(Abs_user);
        }
    }

    free(shell_energies);
    free(Eline);
    free(Eroll);
    free(Emission);
}
