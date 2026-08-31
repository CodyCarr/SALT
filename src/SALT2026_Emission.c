/* SPDX-License-Identifier: BSD-3-Clause */

/*
 * Resonant and fluorescent re-emission from outflow
 *
 * The incident luminosity is tabulated by radial shell in the line-profile
 * driver.  This module applies the bicone projection, escape probabilities,
 * dust, aperture and occultation terms, and attenuation by overlapping
 * transitions.  Emission is accumulated on the observed-velocity grid; the
 * final Maxwellian redistribution is performed by SALT2026_LineProfile.c.
 */

#include <complex.h>
#include <cerf.h>
#include <math.h>
#include <stdlib.h>
#include <gsl/gsl_roots.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_sf_hyperg.h>
#include "salt_grid_config.h"
#include "salt_internal.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Number of blended re-emission lines */
#define MAX_BLENDS 32
/* Number of fluorescent re-emission channels per blended line */
#define MAX_FLUOR  16

enum {
    Nb = SALT_EM_NB,
    Ny = SALT_EM_NY,
    Ns = SALT_EM_NS
};

enum { VOIGT_WOFZ = 0, VOIGT_COLT = 1 };

/* Continued fraction approximation for Voigt profile, see Smith et al. (2015), Appendix A1. */
static inline double colt_approx(double x, double a)
{
    const double A0 = 15.75328153963877;
    const double A1 = 286.9341762324778;
    const double A2 = 19.05706700907019;
    const double A3 = 28.22644017233441;
    const double A4 = 9.526399802414186;
    const double A5 = 35.29217026286130;
    const double A6 = 0.8681020834678775;
    const double B0 = 0.0003300469163682737;
    const double B1 = 0.5403095364583999;
    const double B2 = 2.676724102580895;
    const double B3 = 12.82026082606220;
    const double B4 = 3.21166435627278;
    const double B5 = 32.032981933420;
    const double B6 = 9.0328158696;
    const double B7 = 23.7489999060;
    const double B8 = 1.82106170570;
    const double z = x*x;

    if (z <= 3.0) {
        return exp(-z) * (1.0 - a *
            (A0 + A1/(z - A2 + A3/(z - A4 + A5/(z - A6)))));
    }
    if (z < 25.0) {
        return exp(-z) + a *
            (B0 + B1/(z - B2 + B3/(z + B4 + B5/(z - B6 + B7/(z - B8)))));
    }
    return (a/sqrt(M_PI)) /
        (z - 1.5 - 1.5/(z - 3.5 - 5.0/(z - 5.5)));
}

/* Evaluate the real Voigt/Faddeeva profile with the requested backend. */
static inline double voigt_real(double x, double a, int profile_method)
{
    if (profile_method == VOIGT_COLT)
        return colt_approx(x, a);

    const double complex z = x + a*I;
    return creal(w_of_z(z));
}

__attribute__((constructor))
static void disable_gsl_abort(void)
{
    gsl_set_error_handler_off();
}

typedef struct {
    double A, C, D, E, F, G, H, Ig, K, M, N, O, P, Q, R;
} GeomConst;

typedef struct {
    double wave, fosc, Aul;
    double nu, Del_nu, a1, CS0, shift, tau0;
    int has_fluorescence, n_fluor;
    double fluor_wave[MAX_FLUOR];
    double p_f[MAX_FLUOR];
} BlendLine;

typedef struct {
    double alpha, tau0, y_inf, y_ap;
    double k_dust, delta;
    int APERTURE, RESONANCE, FLUORESCENCE, BLENDING, OCCULTATION;
    int profile_method;
    int GEOM_CASE;
    double c, v_0, y_c, n_0, emitted_wave;
    double G1, G2, G3, G4, G5, G6, G7, G8, G9;
    double G12, G13, G14, G15, G16, G17, G18, G19, G20, G22;
    double yinf_G2, yinf_G6, yinf_G18;
    GeomConst geom;
    int spectrum_uniform;
    double spectrum_dx;
    int n_blends;
    BlendLine blends[MAX_BLENDS];
} EmissionCtx;

/* EmissionCtx stores all geometry, power-law, atomic, blending, and grid
   quantities that remain fixed while observed velocity is varied. */

/* In-place Thomas solve used to construct the incident-spectrum spline. */
static int solve_tridiagonal(int n, double *a, double *b, double *c, double *d)
{
    for (int i = 1; i < n; i++) {
        const double w = a[i] / b[i - 1];
        b[i] -= w * c[i - 1];
        d[i] -= w * d[i - 1];
    }
    d[n - 1] /= b[n - 1];
    for (int i = n - 2; i >= 0; i--)
        d[i] = (d[i] - c[i] * d[i + 1]) / b[i];
    return 1;
}

/* Build second derivatives for a not-a-knot cubic spline. */
static int spline_build_notaknot(const double *x, const double *y, int n, double *m)
{
    if (n < 3) return 0;

    double *a = malloc((size_t)n * sizeof(double));
    double *b = malloc((size_t)n * sizeof(double));
    double *c = malloc((size_t)n * sizeof(double));
    double *d = malloc((size_t)n * sizeof(double));
    double *h = malloc((size_t)(n - 1) * sizeof(double));

    if (!a || !b || !c || !d || !h) {
        free(a); free(b); free(c); free(d); free(h);
        return 0;
    }

    for (int i = 0; i < n - 1; i++) h[i] = x[i + 1] - x[i];

    for (int i = 1; i < n - 1; i++) {
        a[i] = h[i - 1];
        b[i] = 2.0 * (h[i - 1] + h[i]);
        c[i] = h[i];
        d[i] = 6.0 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1]);
    }

    a[0] = 0.0;
    b[0] = h[1];
    c[0] = -(h[0] + h[1]);
    d[0] = 0.0;

    a[n - 1] = -(h[n - 3] + h[n - 2]);
    b[n - 1] = h[n - 3];
    c[n - 1] = 0.0;
    d[n - 1] = 0.0;

    solve_tridiagonal(n, a, b, c, d);

    for (int i = 0; i < n; i++) m[i] = d[i];

    free(a); free(b); free(c); free(d); free(h);
    return 1;
}

/* Evaluate the not-a-knot spline, using constant endpoint extrapolation. */
static double spline_eval_notaknot(const double *x, const double *y, const double *m, int n, double xq)
{
    int lo = 0, hi = n - 1;

    if (xq <= x[0]) lo = 0;
    else if (xq >= x[n - 1]) lo = n - 2;
    else {
        while (hi - lo > 1) {
            const int mid = (lo + hi) >> 1;
            if (x[mid] > xq) hi = mid;
            else lo = mid;
        }
    }

    const double h = x[lo + 1] - x[lo];
    const double A = (x[lo + 1] - xq) / h;
    const double B = (xq - x[lo]) / h;

    return A * y[lo] + B * y[lo + 1]
         + ((A*A*A - A) * m[lo] + (B*B*B - B) * m[lo + 1]) * h*h / 6.0;
}

static inline double safe_hyperg_2F1(double a, double b, double c, double z)
{
    if (!isfinite(z) || z <= -1.0 || z >= 1.0) return 0.0;
    return gsl_sf_hyperg_2F1(a, b, c, z);
}

static inline int nearest_index(
    const double *x_spectrum, int n, double x_target,
    int uniform, double dx
)
{
    if (n <= 1) return 0;

    if (uniform) {
        int idx = (int)llround((x_target - x_spectrum[0]) / dx);
        if (idx < 0) return 0;
        if (idx >= n) return n - 1;
        return idx;
    }

    const int increasing = x_spectrum[n - 1] > x_spectrum[0];

    if (increasing) {
        if (x_target <= x_spectrum[0]) return 0;
        if (x_target >= x_spectrum[n - 1]) return n - 1;
    } else {
        if (x_target >= x_spectrum[0]) return 0;
        if (x_target <= x_spectrum[n - 1]) return n - 1;
    }

    int lo = 0;
    int hi = n - 1;

    while (hi - lo > 1) {
        const int mid = lo + (hi - lo) / 2;
        if (increasing) {
            if (x_spectrum[mid] < x_target)
                lo = mid;
            else
                hi = mid;
        } else {
            if (x_spectrum[mid] > x_target)
                lo = mid;
            else
                hi = mid;
        }
    }

    return (fabs(x_target - x_spectrum[lo])
            <= fabs(x_spectrum[hi] - x_target)) ? lo : hi;
}

static double brent_root(double (*f)(double, void*), void *params, double a, double b)
{
    const double fa = f(a, params);
    const double fb = f(b, params);

    if (!isfinite(fa) || !isfinite(fb) || fa * fb > 0.0)
        return NAN;

    gsl_function F = { .function = f, .params = params };
    gsl_root_fsolver *s = gsl_root_fsolver_alloc(gsl_root_fsolver_brent);

    if (!s) return NAN;
    if (gsl_root_fsolver_set(s, &F, a, b) != GSL_SUCCESS) {
        gsl_root_fsolver_free(s);
        return NAN;
    }

    double r = NAN;
    for (int iter = 0; iter < 100; iter++) {
        if (gsl_root_fsolver_iterate(s) != GSL_SUCCESS) break;
        r = gsl_root_fsolver_root(s);
        a = gsl_root_fsolver_x_lower(s);
        b = gsl_root_fsolver_x_upper(s);
        if (gsl_root_test_interval(a, b, 0.0, 1e-10) != GSL_CONTINUE) break;
    }

    gsl_root_fsolver_free(s);
    return r;
}

typedef struct { double x, G18; } Y1Params;
static double Y1_root_cb(double y, void *pp)
{
    const Y1Params *P = (const Y1Params*)pp;
    return y*y * (1.0 - pow(y, P->G18)) - P->x * P->x;
}
static inline double brent_root_y1(double x, double G18, double y_inf)
{
    Y1Params P = { x, G18 };
    return brent_root(Y1_root_cb, &P, x, y_inf);
}

typedef struct { double x, y_ap, G2, G7; } Y1APParams;
static double Y1AP_root_cb(double y, void *pp)
{
    const Y1APParams *P = (const Y1APParams*)pp;
    return pow(y, P->G7) * P->x * P->x + pow(P->y_ap, P->G2) - pow(y, P->G2);
}
static inline double brent_root_y1ap(double x, double y_ap, double G2, double G7, double y_inf)
{
    Y1APParams P = { x, y_ap, G2, G7 };
    return brent_root(Y1AP_root_cb, &P, x, y_inf);
}

static inline double pow_salt(double x, double exponent)
{
    if (exponent == 0.0) return 1.0;
    if (exponent == 1.0) return x;
    if (exponent == 2.0) return x*x;
    if (exponent == -1.0) return 1.0/x;
    if (exponent == 0.5) return sqrt(x);
    if (exponent == -0.5) return 1.0/sqrt(x);
    if (exponent == -1.5) return 1.0/(x*sqrt(x));
    return pow(x, exponent);
}

/* ------------------------------------------------------------
   Escape optical depth for photons emitted at (x,y).

   Integrates the resonant optical depth encountered by an
   escaping photon after its final scattering until it exits
   the wind. The result is used to attenuate emission by
   subsequent resonant absorption in blended transitions.
   ------------------------------------------------------------ */

/* Optical depth encountered by a photon emitted at shell y as it travels to
   the wind boundary.  Used to attenuate overlapping resonant transitions. */
static void tau_escape(
    const double *y_arr, const double *active_weight, int n,
    double x, double con, double a1, double nu_blend,
    double Del_nu, double c, double y_c, double wavelength, double G1,
    double G2, double G7, double G19, double G20, double G22,
    int OCCULTATION, int profile_method, int Nint, double *tau_out
)
{
    if (Nint < 2) Nint = 2;
    if (Del_nu <= 0.0 || con == 0.0) {
        for (int i = 0; i < n; i++) tau_out[i] = 0.0;
        return;
    }

    for (int i = 0; i < n; i++) {
        if (active_weight != NULL && active_weight[i] == 0.0) {
            tau_out[i] = 0.0;
            continue;
        }
        const double y = y_arr[i];
        const double yG2 = pow_salt(y, G2);
        const double yG7 = pow_salt(y, G7);
        const double radial_offset = yG2 - x*x * yG7;

        double xi_sq = radial_offset;
        if (xi_sq < 0.0) xi_sq = 0.0;
        if (xi_sq > G22) xi_sq = G22;

        const double L = pow(y, G1) * x;
        const double U = sqrt(G22 - xi_sq);

        if (!isfinite(L) || !isfinite(U) || U <= L) {
            tau_out[i] = 0.0;
            continue;
        }

        const double ds = (U - L) / (Nint - 1);
        double tau = 0.0;

        for (int j = 0; j < Nint; j++) {
            const double s = L + ds * j;
            const double w = (j == 0 || j == Nint - 1) ? 0.5 : 1.0;
            const double rad = s*s + radial_offset;

            if (rad > 0.0 && (OCCULTATION || rad >= 1.0)) {
                const double xs = pow_salt(rad, G20) * s;
                const double wave = ((xs - x)/y_c)*wavelength + wavelength;

                if (wave > 0.0 && Del_nu > 0.0) {
                    const double nu = c / (wave * 1e-8);
                    const double a = (nu - nu_blend) / Del_nu;
                    tau += w * con * pow_salt(rad, G19)
                         * voigt_real(a, a1, profile_method);
                }
            }
        }

        tau_out[i] = tau * ds;
    }
}

/* Analytic dust optical depth from the emitting point to the observer-facing
   wind boundary. */
static inline double dust(
    double y, double x, double y_inf, double k_dust, double delta,
    double G3, double G5, double G12, double G13, double G14, double G15, double G16
)
{
    const double xy = x / y;
    double cosT, sinT;

    if (xy <= -1.0) { cosT = -1.0; sinT = 0.0; }
    else if (xy >= 1.0) { cosT = 1.0; sinT = 0.0; }
    else { cosT = xy; sinT = sqrt(1.0 - xy*xy); }

    const double yG3 = pow(y, G3);
    double z = yG3 * sinT;
    if (z < 1e-10) z = 0.0;

    double tau_d;

    if (z == 0.0) {
        tau_d = (delta != 1.0)
              ? k_dust * G5 * (G13 - pow(y, G14))
              : k_dust * G3 * (log(y_inf) - log(y));
    } else {
        const double z2 = z*z;
        const double zG = pow(z, G16);
        const double a1 = sqrt(fmax(0.0, G12*G12 - z2));
        const double a2 = yG3 * cosT;

        double arg1 = -(a1*a1) / z2;
        double arg2 = -(a2*a2) / z2;

        if (arg1 <= -1.0) arg1 = -1.0 + 1e-14;
        if (arg2 <= -1.0) arg2 = -1.0 + 1e-14;

        tau_d = k_dust * ((a1 / zG) * safe_hyperg_2F1(0.5, G15, 1.5, arg1)
                         - (a2 / zG) * safe_hyperg_2F1(0.5, G15, 1.5, arg2));
    }

    return (tau_d > 0.0 && isfinite(tau_d)) ? tau_d : 0.0;
}

static inline double getAsymptotic(double y, double tau, double G4, double G9)
{
    return (3.0 + G9) / (3.0 * tau * pow(y, G4));
}

static inline double getBernoulli3(double y, double tau, double G4, double G9)
{
    const double a = tau * pow(y, G4);

    if (fabs(G9) < 1e-14) {
        const double a2 = a*a;
        return 1.0 / (1.0 + 0.5*a + a2/12.0);
    }

    const double sqrt3 = sqrt(3.0);
    const double complex g9s = csqrt(G9 + 0.0*I);
    const double complex z1 = csqrt(-I*a/sqrt3 + a + 4.0);
    const double complex z2 = csqrt(+I*a/sqrt3 + a + 4.0);
    const double complex d1 = csqrt(12.0 + (3.0 - sqrt3*I)*a);
    const double complex d2 = csqrt(12.0 + (3.0 + sqrt3*I)*a);
    const double complex t1 =
        (sqrt3 + I) * a * catan((2.0*g9s)/z1) / (g9s * d1);
    const double complex t2 =
        (sqrt3 - I) * a * catan((2.0*g9s)/z2) / (g9s * d2);

    return creal(0.5 * (2.0 - t1 - t2));
}

/* Select the Bernoulli or optically thick escape-probability approximation. */
static inline double getBeta(double y, double tau, double G4, double G9, double G17, double CHANGE)
{
    if (CHANGE == 0.0)
        return (G17 > 1.0) ? getAsymptotic(y, tau, G4, G9)
                           : getBernoulli3(y, tau, G4, G9);

    return (y < CHANGE) ? getAsymptotic(y, tau, G4, G9)
                        : getBernoulli3(y, tau, G4, G9);
}

static inline double clampd(double x, double lo, double hi)
{
    return (x < lo) ? lo : (x > hi) ? hi : x;
}

static inline double safe_acos(double x)
{
    return acos(clampd(x, -1.0, 1.0));
}

/* Geometry I: alpha + psi > pi/2 and psi <= alpha. */
static inline double get_GI_scalar(double y, double x, const GeomConst *g, double G1, double G2, double G3, double G7, double G8)
{
    const double lhs = x * pow(y, G1);
    const double yG3 = pow(y, G3);
    const double rad = fmax(0.0, pow(y, G2) - x*x * pow(y, G7));
    const double root = sqrt(rad);

    if (root == 0.0) return 0.0;
    if (lhs > yG3 * g->N) return 1.0;

    const double w = yG3 * g->M - yG3 * g->D / g->C;

    if (lhs > yG3*g->D && lhs > g->C*w && lhs < yG3*g->N)
        return 1.0 - safe_acos((lhs - w*g->C) * g->Q / root) * G8;

    if (lhs > yG3*g->D && lhs < g->C*w && lhs < yG3*g->N) {
        const double DD = yG3*g->M - yG3*g->D/g->E - lhs/g->E;
        const double k = (yG3*g->D + lhs) / g->H;
        const double yA = yG3 * g->A;
        return G8 * atan2(sqrt(fmax(0.0, yA*yA - (yA-k)*(yA-k))), DD);
    }

    if (lhs < yG3*g->D && lhs >= g->C*w && lhs < yG3*g->N) {
        const double yA = yG3 * g->A;
        const double k1 = (yG3*g->D - lhs) / g->H;
        const double DD1 = yG3*g->M - yG3*g->D/g->E + lhs/g->E;
        const double fg_l = G8 * atan2(sqrt(fmax(0.0, yA*yA - (yA-k1)*(yA-k1))), DD1);
        const double fg_u = 1.0 - safe_acos((lhs - w*g->C) * g->Q / root) * G8;
        return fg_u + fg_l;
    }

    {
        const double yA = yG3 * g->A;
        const double k1 = (yG3*g->D - lhs) / g->H;
        const double k2 = (yG3*g->D + lhs) / g->H;
        const double DD1 = yG3*g->M - yG3*g->D/g->E + lhs/g->E;
        const double DD2 = yG3*g->M - yG3*g->D/g->E - lhs/g->E;
        return G8 * atan2(sqrt(fmax(0.0, yA*yA - (yA-k1)*(yA-k1))), DD1)
             + G8 * atan2(sqrt(fmax(0.0, yA*yA - (yA-k2)*(yA-k2))), DD2);
    }
}

/* Geometry II: alpha + psi <= pi/2 and psi <= alpha. */
static inline double get_GII_scalar(double y, double x, const GeomConst *g, double G1, double G2, double G3, double G7, double G8)
{
    const double lhs = x * pow(y, G1);
    const double yG3 = pow(y, G3);
    const double rad = fmax(0.0, pow(y, G2) - x*x * pow(y, G7));
    const double root = sqrt(rad);

    if (root == 0.0) return 0.0;
    if (lhs >= yG3 * g->R) return 1.0;

    if (lhs > yG3*g->Ig + yG3*g->K*g->E && lhs < yG3*g->R) {
        const double n = yG3*g->Ig + yG3*g->O*g->E;
        return 1.0 - safe_acos((lhs - n) * g->F / root) * G8;
    }

    if (lhs > yG3*g->P && lhs < yG3*g->Ig + yG3*g->O*g->E) {
        const double b = (lhs - yG3*g->Ig) / g->H;
        const double yA = yG3 * g->A;
        const double h = yG3*g->O - (lhs - yG3*g->Ig) / g->E;
        return G8 * atan2(sqrt(fmax(0.0, yA*yA - (yA-b)*(yA-b))), h);
    }

    return 0.0;
}

/* Geometry III: alpha + psi > pi/2 and psi > alpha. */
static inline double get_GIII_scalar(double y, double x, const GeomConst *g, double G1, double G3, double G8)
{
    const double lhs = x * pow(y, G1);
    const double yG3 = pow(y, G3);
    const double yA = yG3 * g->A;

    if (lhs >= yG3*g->D && lhs < yG3*g->N) {
        const double DD = yG3*g->M - yG3*g->D/g->E - lhs/g->E;
        const double k = (yG3*g->D + lhs) / g->H;
        return G8 * atan2(sqrt(fmax(0.0, yA*yA - (yA-k)*(yA-k))), DD);
    }

    if (lhs < yG3*g->D) {
        const double ku = (yG3*g->D + lhs) / g->H;
        const double kl = (yG3*g->D - lhs) / g->H;
        const double DDu = yG3*g->M - yG3*g->D/g->E - lhs/g->E;
        const double DDl = yG3*g->M - yG3*g->D/g->E + lhs/g->E;
        return G8 * atan2(sqrt(fmax(0.0, yA*yA - (yA-ku)*(yA-ku))), DDu)
             + G8 * atan2(sqrt(fmax(0.0, yA*yA - (yA-kl)*(yA-kl))), DDl);
    }

    return 0.0;
}

/* Geometry IV: alpha + psi <= pi/2 and psi > alpha. */
static inline double get_GIV_scalar(double y, double x, const GeomConst *g, double G1, double G3, double G8)
{
    const double lhs = x * pow(y, G1);
    const double yG3 = pow(y, G3);

    if (lhs > yG3*g->P && lhs < yG3*g->R) {
        const double p = (lhs - yG3*g->P) / g->H;
        const double yA = yG3 * g->A;
        const double d = yG3*g->O - p*g->G;
        return G8 * atan2(sqrt(fmax(0.0, yA*yA - (yA-p)*(yA-p))), d);
    }

    return 0.0;
}

/* Dispatch the four analytic bicone projections; GEOM_CASE==0 is spherical. */
static inline double geometry_scalar(int GEOM_CASE, double y, double x, const GeomConst *g, double G1, double G2, double G3, double G7, double G8)
{
    switch (GEOM_CASE) {
        case 1: return get_GI_scalar(y, x, g, G1, G2, G3, G7, G8);
        case 2: return get_GII_scalar(y, x, g, G1, G2, G3, G7, G8);
        case 3: return get_GIII_scalar(y, x, g, G1, G3, G8);
        default: return get_GIV_scalar(y, x, g, G1, G3, G8);
    }
}

/* Evaluate the cumulative absorbed luminosity with its physical inner
   boundary F(y=1)=0.  The short linear segment prevents constant spline
   extrapolation from introducing a jump when the emission lower bound crosses
   the launch radius. */
static inline double shell_cdf_eval(
    const double *xk,
    const double *flux,
    const double *mk,
    int nFlux,
    double y
)
{
    if (y <= 1.0) return 0.0;
    if (y < xk[0]) return flux[0] * (y - 1.0) / (xk[0] - 1.0);
    return spline_eval_notaknot(xk, flux, mk, nFlux, y);
}

/* Difference the cumulative absorbed luminosity over bins contained entirely
   within [lower, upper].  Sampling the right edge of each bin avoids the
   singular launch boundary and keeps the result continuous as lower crosses
   y=1 (equivalently, |v_obs|=v_0). */
static void build_shell_grid(
    double lower,
    double upper,
    const double *flux,
    const double *xk,
    const double *mk,
    int nFlux,
    double *y,
    double *shell_E
)
{
    const double dy = (upper - lower) / (double)Ny;
    double previous = shell_cdf_eval(xk, flux, mk, nFlux, lower);

    for (int j = 0; j < Ny; j++) {
        y[j] = lower + dy * (double)(j + 1);
        const double current = shell_cdf_eval(xk, flux, mk, nFlux, y[j]);
        shell_E[j] = current - previous;
        previous = current;
    }
}

/* Blended transitions are processed sequentially in order of
   increasing wavelength. Emission removed by one transition is
   unavailable to subsequent transitions, ensuring photon
   conservation along the escape path.
*/

/* Attenuate shell emission by neighboring transitions and distribute absorbed
   energy among their resonant and fluorescent decay channels. */
static void apply_blending(
    double x,
    const double *y,
    const double *weight,
    double *tau_e,
    const double *blended_p_r,
    const EmissionCtx *C,
    const double *x_spectrum,
    int nSpec,
    double CHANGE,
    double *I_out
)
{
    for (int ib = 0; ib < C->n_blends; ib++) {
        const BlendLine *B = &C->blends[ib];
        if (B->CS0 == 0.0) continue;

        double tau_blend[Ny];
        double absorbed[Ny];

        tau_escape(
            y, weight, Ny, x,
            C->n_0 * B->CS0,
            B->a1, B->nu, B->Del_nu,
            C->c, C->y_c, C->emitted_wave,
            C->G1, C->G2, C->G7,
            C->G19, C->G20, C->G22,
            C->OCCULTATION, C->profile_method,
            Ns,
            tau_blend
        );

        const int idx = nearest_index(
            x_spectrum, nSpec, x - B->shift,
            C->spectrum_uniform, C->spectrum_dx
        );

        for (int j = 0; j < Ny; j++) {
            if (weight[j] == 0.0) {
                absorbed[j] = 0.0;
                continue;
            }
            absorbed[j] =
                weight[j] * exp(-tau_e[j]) * (1.0 - exp(-tau_blend[j]));

            tau_e[j] += tau_blend[j];
        }

        if (B->has_fluorescence) {
            double I_res = 0.0;
            double I_f[MAX_FLUOR] = {0.0};

            for (int j = 0; j < Ny; j++) {
                if (absorbed[j] == 0.0) continue;
                const double BB = getBeta(y[j], B->tau0, C->G4, C->G9, C->G17, CHANGE);
                const double den = 1.0 - blended_p_r[ib] * (1.0 - BB);

                if (den != 0.0) {
                    const double absorbed_over_den = absorbed[j] / den;
                    I_res += absorbed_over_den * BB * blended_p_r[ib];
                    for (int jf = 0; jf < B->n_fluor; jf++)
                        I_f[jf] += absorbed_over_den * B->p_f[jf];
                }
            }

            I_out[idx] += I_res;

            for (int jf = 0; jf < B->n_fluor; jf++) {
                const double fshift =
                    (C->c * (C->emitted_wave - B->fluor_wave[jf]) / C->emitted_wave) / C->v_0;

                const int idxf = nearest_index(
                    x_spectrum, nSpec, x - fshift,
                    C->spectrum_uniform, C->spectrum_dx
                );

                I_out[idxf] += I_f[jf];
            }
        } else {
            for (int j = 0; j < Ny; j++)
                I_out[idx] += absorbed[j];
        }
    }
}

/* ------------------------------------------------------------
   Emission at dimensionless velocity x.

   Computes resonant and fluorescent emission from the wind.
   For x > 0, the approaching-side limits are used. For x < 0,
   the receding-side limits are used, including occultation when
   enabled. If line blending is enabled, photons removed by one
   transition are propagated through subsequent blended transitions
   before escaping.
   ------------------------------------------------------------ */

/* Integrate shell contributions at one observed velocity, applying geometry,
   aperture, occultation, escape probability, dust, and optional blending. */
static void Emission_Integral(
    double x, int red, int output_index,
    double p_f, double p_r, const double *blended_p_r,
    const EmissionCtx *C, const double *flux, const double *xk, const double *mk,
    int nFlux, const double *x_spectrum, int nSpec, double CHANGE, double *I_out
)
{
    const double ax = fabs(x);
    double lower = red ? fmax(ax, 1.0) : fmax(x, 1.0);
    double upper = C->y_inf;

    if (red && C->OCCULTATION) {
        const double cond = C->y_inf*C->y_inf * (1.0 - C->yinf_G18) - ax*ax;
        if (cond > 0.0) {
            const double r = brent_root_y1(ax, C->G18, C->y_inf);
            if (isfinite(r)) lower = r;
        }
    }

    if (C->APERTURE) {
        const double cond = C->yinf_G6 * x*x + pow(C->y_ap, C->G2) - C->yinf_G2;
        if (cond < 0.0) {
            const double r = brent_root_y1ap(ax, C->y_ap, C->G2, C->G7, C->y_inf);
            if (isfinite(r)) upper = r;
        }
    }

    /* The geometry inequalities share a factor y^G3, while
       x*y^G1/y^G3 = x/y for every gamma.  Clip exact zero-coverage shell
       ranges before building the fixed grid. */
    if ((C->GEOM_CASE == 2 || C->GEOM_CASE == 4) && C->geom.P > 0.0) {
        upper = fmin(upper, ax / C->geom.P);
    } else if (C->GEOM_CASE == 3 && C->geom.N > 0.0) {
        lower = fmax(lower, ax / C->geom.N);
    }

    if (upper <= lower) return;

    double y[Ny], shell_E[Ny], weight[Ny], tau_e[Ny];

    build_shell_grid(lower, upper, flux, xk, mk, nFlux, y, shell_E);

    int any_weight = 0;
    for (int j = 0; j < Ny; j++) {
        const double yy = y[j];
        tau_e[j] = 0.0;

        if (shell_E[j] == 0.0) {
            weight[j] = 0.0;
            continue;
        }

        const double fg = geometry_scalar(C->GEOM_CASE, yy, red ? ax : x,
                                          &C->geom, C->G1, C->G2, C->G3, C->G7, C->G8);
        if (fg == 0.0) {
            weight[j] = 0.0;
            continue;
        }

        double fac = 1.0;
        if (C->RESONANCE || C->FLUORESCENCE) {
            const double BB = getBeta(yy, C->tau0, C->G4, C->G9, C->G17, CHANGE);
            const double den = 1.0 - p_r * (1.0 - BB);
            fac = (den != 0.0) ? (C->RESONANCE ? BB * p_r / den : p_f / den) : 0.0;
        }
        if (fac == 0.0) {
            weight[j] = 0.0;
            continue;
        }

        const double tau_d = (C->k_dust == 0.0) ? 0.0
            : dust(yy, x, C->y_inf, C->k_dust, C->delta,
                   C->G3, C->G5, C->G12, C->G13, C->G14, C->G15, C->G16);
        weight[j] = exp(-tau_d) * fac * fg * shell_E[j] / (2.0 * yy);
        any_weight |= weight[j] != 0.0;
    }

    if (!any_weight) return;

    const int idx_x = output_index;

    if (!C->BLENDING) {
        for (int j = 0; j < Ny; j++) I_out[idx_x] += weight[j];
        return;
    }

    apply_blending(x, y, weight, tau_e, blended_p_r, C, x_spectrum, nSpec, CHANGE, I_out);

    for (int j = 0; j < Ny; j++)
        I_out[idx_x] += weight[j] * exp(-tau_e[j]);

}

/* Per-velocity worker kept separate for OpenMP and serial execution paths. */
static void emission_worker_x(
    double x, int output_index,
    double p_f, double p_r, const double *blended_p_r,
    const EmissionCtx *C, const double *flux, const double *xk, const double *mk,
    int nFlux, const double *x_spectrum, int nSpec, double CHANGE, double *E_out
)
{
    if (x == 0.0) x = 1e-10;
    if (fabs(x) > fabs(C->y_inf) || C->alpha == 0.0 || C->tau0 == 0.0) return;

    Emission_Integral(x, x < 0.0, output_index,
                      p_f, p_r, blended_p_r, C, flux, xk, mk,
                      nFlux, x_spectrum, nSpec, CHANGE, E_out);
}

/*
 * Compute one emitted channel over v_obs.
 *
 * shell_luminosity is produced by the line-profile driver from the absorbed
 * continuum.  The output is an additive, continuum-normalized emission
 * profile before the final Maxwellian redistribution performed by the driver.
 */
void computeEM_vector(
    double wavelength, double emitted_wave,
    const double *blended_waves, const double *blended_osc_strs,
    const double *blended_abs_ein, const int *blended_fluor,
    const double *blended_p_r, const double *blended_p_f,
    const double *blended_fluor_waves, const int *n_fluor_each, int n_blends,
    double oscillator_strength, double einstein_coefficient, double lambda_ref,
    const double *v_obs, int n, const double *Normalized_Flux, int nFlux,
    double alpha, double psi, double gamma, double tau,
    double v_0, double v_w, double v_ap, double v_b,
    double f_c, double k_dust, double delta,
    int APERTURE, int RESONANCE, int FLUORESCENCE, int BLENDING, int OCCULTATION,
    int profile_method, double p_r, double p_f, double *E_out
)
{
    (void)einstein_coefficient;
    (void)lambda_ref;

    for (int i = 0; i < n; i++) E_out[i] = 0.0;
    if (n <= 0 || nFlux < 3 || v_0 == 0.0 || v_ap == 0.0 || delta == 0.0) return;
    if (alpha == 0.0 || tau == 0.0 || oscillator_strength == 0.0 || f_c == 0.0) return;
    if ((RESONANCE && p_r == 0.0) || (!RESONANCE && FLUORESCENCE && p_f == 0.0)) return;

    const double c = 2.99792458e10;
    const double v0_cms = v_0 * 1e5;
    const double y_inf = v_w / v_0;

    EmissionCtx C = {0};

    /* Precompute exponents and geometry constants used throughout
   the emission calculation. */

    C.alpha = alpha;
    C.tau0 = wavelength * oscillator_strength * tau;
    C.y_inf = y_inf; C.y_ap = v_ap / v_0;
    C.k_dust = k_dust; C.delta = delta;
    C.APERTURE = APERTURE != 0; C.RESONANCE = RESONANCE != 0;
    C.FLUORESCENCE = FLUORESCENCE != 0; C.BLENDING = BLENDING != 0;
    C.OCCULTATION = OCCULTATION != 0;
    C.profile_method = (profile_method == VOIGT_COLT) ? VOIGT_COLT : VOIGT_WOFZ;
    C.c = c; C.v_0 = v0_cms;
    C.y_c = c / fabs(v0_cms);
    C.n_0 = tau * pow(0.0265397, -1.0) * fabs(v0_cms) * 1e8;
    C.emitted_wave = emitted_wave;

    C.G1 = (1.0 - gamma) / gamma;
    C.G2 = 2.0 / gamma;
    C.G3 = 1.0 / gamma;
    C.G4 = (1.0 - (delta + gamma)) / gamma;
    C.G5 = (delta != 1.0) ? 1.0 / (1.0 - delta) : 1.0;
    C.G6 = (gamma - 1.0) / gamma;
    C.G7 = 2.0 * (1.0 - gamma) / gamma;
    C.G8 = 1.0 / M_PI;
    C.G9 = gamma - 1.0;
    C.G12 = pow(y_inf, 1.0 / gamma);
    C.G13 = pow(y_inf, (1.0 - delta) / gamma);
    C.G14 = (1.0 - delta) / gamma;
    C.G15 = delta / 2.0;
    C.G16 = 2.0 + gamma;
    C.G17 = C.tau0 * pow(y_inf, C.G4);
    C.G18 = -2.0 / gamma;
    C.G19 = -delta / 2.0;
    C.G20 = (gamma - 1.0) / 2.0;
    C.G22 = pow(y_inf, C.G2);

    C.yinf_G2 = pow(y_inf, C.G2);
    C.yinf_G6 = pow(y_inf, C.G6);
    C.yinf_G18 = pow(y_inf, C.G18);

    C.geom.A = sin(alpha);
    C.geom.C = tan(alpha - fabs(psi - alpha));
    C.geom.D = sin(psi + alpha - M_PI/2.0);
    C.geom.E = tan(psi);
    C.geom.F = tan(M_PI/2.0 - psi);
    C.geom.G = cos(psi);
    C.geom.H = sin(psi);
    C.geom.Ig = cos(psi + alpha);
    C.geom.K = sin(psi + alpha);
    C.geom.M = cos(psi + alpha - M_PI/2.0);
    C.geom.N = cos(fabs(psi - alpha));
    C.geom.O = sin(alpha + psi);
    C.geom.P = cos(alpha + psi);
    C.geom.Q = tan(M_PI/2.0 - alpha + fabs(psi - alpha));
    C.geom.R = cos(psi - alpha);

    /* Select the analytic projection from the opening angle alpha and the
       orientation angle psi.  Boundary equalities follow the cases below. */
    if (alpha + psi > M_PI/2.0 && psi - alpha <= 0.0) C.GEOM_CASE = 1;
    else if (alpha + psi <= M_PI/2.0 && psi - alpha <= 0.0) C.GEOM_CASE = 2;
    else if (alpha + psi > M_PI/2.0 && psi - alpha > 0.0) C.GEOM_CASE = 3;
    else C.GEOM_CASE = 4;

    C.n_blends = (C.BLENDING && n_blends > 0) ? ((n_blends < MAX_BLENDS) ? n_blends : MAX_BLENDS) : 0;

    for (int i = 0; i < C.n_blends; i++) {
        C.blends[i].wave = blended_waves[i];
        C.blends[i].fosc = blended_osc_strs[i];
        C.blends[i].Aul = blended_abs_ein[i];
        C.blends[i].has_fluorescence = blended_fluor[i] != 0;
        C.blends[i].n_fluor = n_fluor_each[i] < MAX_FLUOR ? n_fluor_each[i] : MAX_FLUOR;

        C.blends[i].nu = c / (C.blends[i].wave * 1e-8);
        C.blends[i].Del_nu = fabs(v_b) * 1e5 / c * C.blends[i].nu;
        C.blends[i].a1 = (C.blends[i].Del_nu > 0.0)
            ? C.blends[i].Aul / (4.0 * M_PI * C.blends[i].Del_nu) : 0.0;
        C.blends[i].CS0 = (C.blends[i].Del_nu > 0.0)
            ? 0.0265397 / sqrt(M_PI) * C.blends[i].fosc / C.blends[i].Del_nu : 0.0;
        C.blends[i].shift =
            (c * (emitted_wave - C.blends[i].wave) / emitted_wave) / v0_cms;
        C.blends[i].tau0 = C.blends[i].fosc * C.blends[i].wave * tau;

        for (int j = 0; j < C.blends[i].n_fluor; j++) {
            const int flat = i * MAX_FLUOR + j;
            C.blends[i].fluor_wave[j] = blended_fluor_waves[flat];
            C.blends[i].p_f[j] = blended_p_f[flat];
        }
    }

    /* Locate the shell where the intermediate- and high-optical-depth escape
       probability approximations should switch. */
    double CHANGE = 0.0;

    for (int i = 1; i < Nb - 1; i++) {
        const double y  = 1.0 + (y_inf - 1.0) * i / (double)(Nb - 1);
        const double yL = 1.0 + (y_inf - 1.0) * (i - 1) / (double)(Nb - 1);
        const double yR = 1.0 + (y_inf - 1.0) * (i + 1) / (double)(Nb - 1);
        const double d  = getBernoulli3(y,  C.tau0, C.G4, C.G9) - getAsymptotic(y,  C.tau0, C.G4, C.G9);
        const double dL = getBernoulli3(yL, C.tau0, C.G4, C.G9) - getAsymptotic(yL, C.tau0, C.G4, C.G9);
        const double dR = getBernoulli3(yR, C.tau0, C.G4, C.G9) - getAsymptotic(yR, C.tau0, C.G4, C.G9);

        if (d > dL && d > dR) {
            CHANGE = y;
            break;
        }
    }

    /* Interpolate the absorbed shell luminosity smoothly because the outer
       emission quadrature samples between the driver's discrete shells. */
    double *xk = malloc((size_t)nFlux * sizeof(double));
    double *flux = malloc((size_t)nFlux * sizeof(double));
    double *mk = malloc((size_t)nFlux * sizeof(double));
    double *xspec = malloc((size_t)n * sizeof(double));

    if (!xk || !flux || !mk || !xspec) {
        free(xk); free(flux); free(mk); free(xspec);
        return;
    }

    for (int i = 0; i < nFlux; i++)
        xk[i] = 1.0 + (y_inf - 1.0) * i / (double)(nFlux - 1);

    xk[0] += y_inf / (double)SALT_EM_FIRST_FLUX_OFFSET;

    for (int i = 0; i < nFlux; i++)
        flux[i] = Normalized_Flux[i] / fabs(v_0);

    if (!spline_build_notaknot(xk, flux, nFlux, mk)) {
        free(xk); free(flux); free(mk); free(xspec);
        return;
    }

    for (int i = 0; i < n; i++)
        xspec[i] = v_obs[i] / v_0;

    C.spectrum_uniform = n < 3;
    C.spectrum_dx = n > 1 ? xspec[1] - xspec[0] : 0.0;

    if (n >= 3 && C.spectrum_dx != 0.0) {
        C.spectrum_uniform = 1;
        const double tol = 32.0 * 2.2204460492503131e-16
                         * fmax(1.0, fabs(xspec[n - 1]));
        for (int i = 2; i < n; i++) {
            const double expected = xspec[0] + C.spectrum_dx * (double)i;
            if (fabs(xspec[i] - expected) > tol) {
                C.spectrum_uniform = 0;
                break;
            }
        }
    }

    int nthreads = 1;
    #ifdef _OPENMP
    nthreads = omp_get_max_threads();
    #endif

    /* Blended fluorescent channels may deposit power in velocity bins other
       than the worker's input bin.  Thread-private spectra avoid atomics and
       are reduced after both red- and blue-side passes. */
    double *thread_out = calloc((size_t)nthreads * (size_t)n, sizeof(double));

    if (thread_out) {
        for (int pass = 0; pass < 2; pass++) {
            #ifdef _OPENMP
            #pragma omp parallel for schedule(static, 1)
            #endif
            for (int i = 0; i < n; i++) {
                if ((pass == 0 && v_obs[i] >= 0.0) ||
                    (pass == 1 && v_obs[i] < 0.0))
                    continue;

                int tid = 0;
                #ifdef _OPENMP
                tid = omp_get_thread_num();
                #endif

                emission_worker_x(v_obs[i] / v_0, i,
                                  p_f, p_r, blended_p_r,
                                  &C, flux, xk, mk, nFlux, xspec, n, CHANGE,
                                  thread_out + (size_t)tid * (size_t)n);
            }
        }

        for (int t = 0; t < nthreads; t++) {
            const double *local = thread_out + (size_t)t * (size_t)n;
            for (int i = 0; i < n; i++)
                E_out[i] += local[i];
        }
        free(thread_out);
    } else {
        for (int pass = 0; pass < 2; pass++) {
            for (int i = 0; i < n; i++) {
                if ((pass == 0 && v_obs[i] >= 0.0) ||
                    (pass == 1 && v_obs[i] < 0.0))
                    continue;

                emission_worker_x(v_obs[i] / v_0, i,
                                  p_f, p_r, blended_p_r,
                                  &C, flux, xk, mk, nFlux, xspec, n, CHANGE,
                                  E_out);
            }
        }
    }

    /* Fixes x == 0. Use the resolved two-sided grid limit. */
    for (int i = 1; i + 1 < n; ++i) {
        if (v_obs[i] == 0.0) {
            const double width = v_obs[i + 1] - v_obs[i - 1];
            E_out[i] = width != 0.0
                     ? (E_out[i - 1] * (v_obs[i + 1] - v_obs[i]) +
                        E_out[i + 1] * (v_obs[i] - v_obs[i - 1])) / width
                     : 0.5 * (E_out[i - 1] + E_out[i + 1]);
        }
    }

    free(xk);
    free(flux);
    free(mk);
    free(xspec);
}
