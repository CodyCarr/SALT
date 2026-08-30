/* SPDX-License-Identifier: BSD-3-Clause */

/*
 * Continuum absorption by an accelerating spherical or biconical SALT wind.
 *
 * For v_b > 0 the optical depth is integrated through the finite resonance
 * volume using a Voigt cross section.  For v_b == 0 the code evaluates the
 * original Sobolev-limit geometry.  Velocities are normalized by v_0 inside
 * the kernel; wavelengths are in Angstrom and velocities supplied by callers
 * are in km s^-1.
 */

#include <math.h>
#include <complex.h>
#include <cerf.h>
#include <stdlib.h>
#include <gsl/gsl_roots.h>
#include <gsl/gsl_errno.h>
#include "salt_grid_config.h"
#include "salt_internal.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#ifndef M_PI
#define M_PI 3.141592653589793238462643383279502884
#endif

enum {
    Ns = SALT_ABS_NS,
    Nphi = SALT_ABS_NPHI,
    Ny = SALT_ABS_NY,
    NySobolev = SALT_ABS_NY_SOBOLEV
};
static const double disc_eps = 1e-14;

enum { VOIGT_WOFZ = 0, VOIGT_COLT = 1 };

/* Geometry-only quantities reused at every observed velocity. */
typedef struct {
    double h[Ny];
    double h2[Ny];
    double L[Ny];
    double U[Ny];
    double ds[Ny];
    unsigned char valid[Ny];
    double density[Ny * Ns];
    double xs[Ny * Ns];
} TurbulentGrid;

static double phi_sin[Nphi];
static double phi_cos[Nphi];

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
    const double dphi = 2.0 * M_PI / (double)Nphi;
    for (int k = 0; k < Nphi; k++) {
        const double phi = -M_PI + dphi * (double)k;
        phi_sin[k] = sin(phi);
        phi_cos[k] = cos(phi);
    }
}

typedef struct { double absx, y_ap, G2, G5; } Y1APParams;
typedef struct { double absx, G9; } Y1Params;

static double y1ap_root_f(double y, void *params)
{
    const Y1APParams *P = (const Y1APParams*)params;
    return pow(y, P->G5) * P->absx * P->absx
         + pow(P->y_ap, P->G2)
         - pow(y, P->G2);
}

static double y1_root_f(double y, void *params)
{
    const Y1Params *P = (const Y1Params*)params;
    return y*y*(1.0 - pow(y, P->G9)) - P->absx*P->absx;
}

static int brent_root(
    double (*f)(double, void*),
    void *params,
    double lower,
    double upper,
    double *root
)
{
    if (!(upper > lower)) return 0;

    const double fl = f(lower, params);
    const double fu = f(upper, params);

    if (!isfinite(fl) || !isfinite(fu)) return 0;
    if (fl == 0.0) { *root = lower; return 1; }
    if (fu == 0.0) { *root = upper; return 1; }
    if (fl * fu > 0.0) return 0;

    gsl_function F;
    F.function = f;
    F.params = params;

    gsl_root_fsolver *s = gsl_root_fsolver_alloc(gsl_root_fsolver_brent);
    if (!s) return 0;

    if (gsl_root_fsolver_set(s, &F, lower, upper) != GSL_SUCCESS) {
        gsl_root_fsolver_free(s);
        return 0;
    }

    double r = lower;
    for (int iter = 0; iter < 100; iter++) {
        if (gsl_root_fsolver_iterate(s) != GSL_SUCCESS) break;

        r = gsl_root_fsolver_root(s);
        lower = gsl_root_fsolver_x_lower(s);
        upper = gsl_root_fsolver_x_upper(s);

        if (gsl_root_test_interval(lower, upper, 0.0, 1e-10) != GSL_CONTINUE)
            break;
    }

    gsl_root_fsolver_free(s);
    *root = r;
    return isfinite(r);
}

static inline double positive_floor(double v, double eps)
{
    return (v < eps) ? eps : v;
}

static double simpson_samples(const double *f, int n, double dx)
{
    if (n < 3 || (n % 2) == 0) return 0.0;

    double sum = f[0] + f[n - 1];
    for (int i = 1; i < n - 1; i++)
        sum += (i & 1 ? 4.0 : 2.0) * f[i];

    return sum * dx / 3.0;
}

/* ---------------- Resonant integrand ---------------- */
static inline double Resonant_OD_Integrand_fast(
    double s,
    double y,
    double yG2,
    double yG5,
    double x,
    double x2,
    double a1,
    double nu_lu,
    double Del_nu,
    double c,
    double y_c,
    double lambda_lu,
    double G11,
    double G12,
    int profile_method
)
{
    (void)y;

    const double base = s*s + yG2 - x2*yG5;
    if (base <= 0.0) return 0.0;

    const double xs = pow(base, G12) * s;

    /* Comoving-frame wavelength sampled by gas at position s. */
    const double wave = ((xs - x)/y_c)*lambda_lu + lambda_lu;

    if (wave <= 0.0 || Del_nu <= 0.0) return 0.0;

    const double nu = c / wave;
    const double a  = (nu - nu_lu) / Del_nu;

    const double w = voigt_real(a, a1, profile_method);

    return pow(base, G11) * w;
}

static inline double Resonant_OD_Integrand_cached(
    double density,
    double xs,
    double x,
    double a1,
    double nu_lu,
    double Del_nu,
    double c,
    double y_c,
    double lambda_lu,
    int profile_method
)
{
    const double wave = ((xs - x)/y_c)*lambda_lu + lambda_lu;
    if (wave <= 0.0 || Del_nu <= 0.0) return 0.0;

    const double a = (c/wave - nu_lu) / Del_nu;
    return density * voigt_real(a, a1, profile_method);
}

static void build_turbulent_grid(
    TurbulentGrid *grid,
    double lower,
    double upper,
    double SA,
    double G11,
    double G12
)
{
    const double dh = (upper - lower) / (Ny - 1);

    for (int i = 0; i < Ny; i++) {
        double hi = lower + (double)i * dh;
        if (i == 0) hi += 1e-6;

        const double h2 = hi * hi;
        grid->h[i] = hi;
        grid->h2[i] = h2;
        grid->valid[i] = 0;

        if (h2 >= 1.0 || h2 >= SA) continue;

        const double L = sqrt(fmax(0.0, 1.0 - h2));
        const double U = sqrt(fmax(0.0, SA - h2));
        if (!(U > L)) continue;

        const double ds = (U - L) / (Ns - 1);
        grid->L[i] = L;
        grid->U[i] = U;
        grid->ds[i] = ds;
        grid->valid[i] = 1;

        const size_t off = (size_t)i * Ns;
        for (int j = 0; j < Ns; j++) {
            const double s = L + ds * (double)j;
            const double base = s*s + h2;
            grid->density[off + (size_t)j] = pow(base, G11);
            grid->xs[off + (size_t)j] = pow(base, G12) * s;
        }
    }
}

/* ================== Orientation I (alpha >= pi/2) ================== */
static void Orientation_I(
    const double *h,
    const double *hG2,
    const double *hG5,
    int n,
    double x,
    double con,
    double a1,
    double nu_lu,
    double Del_nu,
    double c,
    double y_c,
    double lambda_lu,
    double SA,
    double SB,
    double G11,
    double G12,
    int profile_method,
    int Ns_local,
    double *out,
    const TurbulentGrid *grid
)
{
    (void)hG5;

    if (Ns_local < 3) Ns_local = 3;

    for (int i = 0; i < n; i++) {

        const double hi  = h[i];
        const double h2  = hG2[i];

        if (grid && grid->valid[i]) {
            const double ds = grid->ds[i];
            const size_t off = (size_t)i * Ns;
            double tau = 0.0;

            for (int j = 0; j < Ns; j++) {
                const double weight = (j == 0 || j == Ns - 1) ? 0.5 : 1.0;
                tau += weight * con * Resonant_OD_Integrand_cached(
                    grid->density[off + (size_t)j],
                    grid->xs[off + (size_t)j], x, a1, nu_lu, Del_nu,
                    c, y_c, lambda_lu, profile_method
                );
            }
            out[i] = SB * (1.0 - exp(-tau * ds));
            continue;
        }

        if (h2 >= 1.0 || h2 >= SA) {
            out[i] = 0.0;
            continue;
        }

        const double L = sqrt(fmax(0.0, 1.0 - h2));
        const double U = sqrt(fmax(0.0, SA  - h2));

        if (!(U > L)) {
            out[i] = 0.0;
            continue;
        }

        const double ds = (U - L) / (Ns_local - 1);

        double tau = 0.0;

        for (int j = 0; j < Ns_local; j++) {
            const double s = L + ds * (double)j;

            const double weight =
                (j == 0 || j == Ns_local - 1) ? 0.5 : 1.0;

            tau += weight * con * Resonant_OD_Integrand_fast(
                s,
                hi,
                h2,
                0.0,
                x,
                0.0,
                a1,
                nu_lu,
                Del_nu,
                c,
                y_c,
                lambda_lu,
                G11,
                G12,
                profile_method
            );
        }

        tau *= ds;

        out[i] = SB * (1.0 - exp(-tau));
    }
}

/* ================== Orientation II (psi > alpha) ================== */
static void Orientation_II(
    const double *h,
    const double *hG2,
    const double *hG5,
    int n,
    double x,
    double con,
    double a1,
    double nu_lu,
    double Del_nu,
    double c,
    double y_c,
    double lambda_lu,
    double SA,
    double SB,
    double SC,
    double SD,
    double SE,
    double G11,
    double G12,
    int profile_method,
    int Ns_local,
    int Nphi_local,
    double *out,
    const TurbulentGrid *grid
)
{
    (void)hG5;

    if (Ns_local < 3) Ns_local = 3;
    if (Nphi_local < 8) Nphi_local = 8;

    const double dphi = 2.0 * M_PI / (double)Nphi_local;

    for (int i = 0; i < n; i++) {

        const double hi = h[i];
        const double h2 = hG2[i];

        if (h2 <= 0.0 || h2 >= 1.0 || h2 >= SA) {
            out[i] = 0.0;
            continue;
        }

        const double L = sqrt(fmax(0.0, 1.0 - h2));
        const double U = sqrt(fmax(0.0, SA  - h2));

        if (!(U > L)) {
            out[i] = 0.0;
            continue;
        }

        const double ds = (U - L) / (Ns_local - 1);
        const double inv_ds = 1.0 / ds;

        double dvec[Ns_local];

        double s = L;
        const size_t off = (size_t)i * Ns;
        double f_prev = grid ? con * Resonant_OD_Integrand_cached(
            grid->density[off], grid->xs[off], x, a1, nu_lu, Del_nu,
            c, y_c, lambda_lu, profile_method
        ) : con * Resonant_OD_Integrand_fast(
            s, hi, h2, 0.0, x, 0.0, a1, nu_lu, Del_nu, c, y_c,
            lambda_lu, G11, G12, profile_method
        );

        dvec[0] = 0.0;

        for (int j = 1; j < Ns_local; j++) {
            s = L + ds * (double)j;

            const double f_cur = grid ? con * Resonant_OD_Integrand_cached(
                grid->density[off + (size_t)j], grid->xs[off + (size_t)j],
                x, a1, nu_lu, Del_nu, c, y_c, lambda_lu, profile_method
            ) : con * Resonant_OD_Integrand_fast(
                s, hi, h2, 0.0, x, 0.0,
                a1, nu_lu, Del_nu, c, y_c, lambda_lu,
                G11, G12, profile_method
            );

            dvec[j] = dvec[j-1] + 0.5 * (f_prev + f_cur) * ds;
            f_prev = f_cur;
        }

        double sum_abs = 0.0;

        for (int k = 0; k < Nphi_local; k++) {

            const double sv = (Nphi_local == Nphi) ? phi_sin[k]
                                                    : sin(-M_PI + dphi*(double)k);
            const double cv = (Nphi_local == Nphi) ? phi_cos[k]
                                                    : cos(-M_PI + dphi*(double)k);

            const double BB  = hi * sv * SC;
            const double CCc = h2 * (sv*sv*SD - cv*cv*SE);

            const double disc = BB*BB - 4.0*SB*CCc;
            if (disc <= disc_eps) continue;

            const double sqrt_disc = sqrt(disc);
            const double denom = 2.0 * SB;

            double r1 = (-BB - sqrt_disc) / denom;
            double r2 = (-BB + sqrt_disc) / denom;

            double s_in  = fmin(r1, r2);
            double s_out = fmax(r1, r2);

            if (s_in  < L) s_in  = L;
            if (s_out > U) s_out = U;
            if (s_out <= s_in) continue;

            double u = (s_in - L) * inv_ds;
            int j = (int)u;
            if (j < 0) j = 0;
            if (j > Ns_local - 2) j = Ns_local - 2;

            double t = u - (double)j;
            const double F_in = dvec[j] * (1.0 - t) + dvec[j+1] * t;

            u = (s_out - L) * inv_ds;
            j = (int)u;
            if (j < 0) j = 0;
            if (j > Ns_local - 2) j = Ns_local - 2;

            t = u - (double)j;
            const double F_out = dvec[j] * (1.0 - t) + dvec[j+1] * t;

            const double tau_seg = F_out - F_in;

            if (tau_seg > 0.0)
                sum_abs += 1.0 - exp(-tau_seg);
        }

        out[i] = sum_abs * dphi;
    }
}

/* ==================== Orientation III (psi < alpha) ====================== */
static void Orientation_III(
    const double *h,
    const double *hG2,
    const double *hG5,
    int n,
    double x,
    double con,
    double a1,
    double nu_lu,
    double Del_nu,
    double c,
    double y_c,
    double lambda_lu,
    double SA,
    double Ap,
    double Bp,
    double Cp,
    double Cpp,
    double G11,
    double G12,
    int profile_method,
    int Ns_local,
    int Nphi_local,
    double *out,
    const TurbulentGrid *grid
)
{
    (void)hG5;

    if (Ns_local < 3) Ns_local = 3;
    if (Nphi_local < 8) Nphi_local = 8;

    const double dphi = 2.0 * M_PI / (double)Nphi_local;

    for (int i = 0; i < n; i++) {

        const double hi = h[i];
        const double h2 = hG2[i];

        if (h2 <= 0.0 || h2 >= 1.0 || h2 >= SA) {
            out[i] = 0.0;
            continue;
        }

        const double L = sqrt(fmax(0.0, 1.0 - h2));
        const double U = sqrt(fmax(0.0, SA  - h2));

        if (!(U > L)) {
            out[i] = 0.0;
            continue;
        }

        const double ds = (U - L) / (Ns_local - 1);
        const double inv_ds = 1.0 / ds;

        double dvec[Ns_local];

        double s = L;
        const size_t off = (size_t)i * Ns;
        double f_prev = grid ? con * Resonant_OD_Integrand_cached(
            grid->density[off], grid->xs[off], x, a1, nu_lu, Del_nu,
            c, y_c, lambda_lu, profile_method
        ) : con * Resonant_OD_Integrand_fast(
            s, hi, h2, 0.0, x, 0.0, a1, nu_lu, Del_nu, c, y_c,
            lambda_lu, G11, G12, profile_method
        );

        dvec[0] = 0.0;

        for (int j = 1; j < Ns_local; j++) {
            s = L + ds * (double)j;

            const double f_cur = grid ? con * Resonant_OD_Integrand_cached(
                grid->density[off + (size_t)j], grid->xs[off + (size_t)j],
                x, a1, nu_lu, Del_nu, c, y_c, lambda_lu, profile_method
            ) : con * Resonant_OD_Integrand_fast(
                s, hi, h2, 0.0, x, 0.0,
                a1, nu_lu, Del_nu, c, y_c, lambda_lu,
                G11, G12, profile_method
            );

            dvec[j] = dvec[j-1] + 0.5 * (f_prev + f_cur) * ds;
            f_prev = f_cur;
        }

        const double tauU = dvec[Ns_local - 1];

        double sum_abs = 0.0;

        for (int k = 0; k < Nphi_local; k++) {

            const double sv = (Nphi_local == Nphi) ? phi_sin[k]
                                                    : sin(-M_PI + dphi*(double)k);
            const double cv = (Nphi_local == Nphi) ? phi_cos[k]
                                                    : cos(-M_PI + dphi*(double)k);

            const double B = hi * sv * Bp;
            const double C = h2 * (sv*sv*Cp - cv*cv*Cpp);

            const double disc = B*B - 4.0 * Ap * C;
            if (disc <= disc_eps) continue;

            const double sqrt_disc = sqrt(disc);
            const double denom = 2.0 * Ap;

            double r1 = (-B - sqrt_disc) / denom;
            double r2 = (-B + sqrt_disc) / denom;

            double s_lo = fmin(r1, r2);
            double s_hi = fmax(r1, r2);

            if (s_lo < L) s_lo = L;
            if (s_hi > U) s_hi = U;

            double tau_seg = 0.0;

            if (s_lo > L) {
                double u = (s_lo - L) * inv_ds;
                int j = (int)u;
                if (j < 0) j = 0;
                if (j > Ns_local - 2) j = Ns_local - 2;

                const double t = u - (double)j;
                const double F = dvec[j] * (1.0 - t) + dvec[j+1] * t;

                tau_seg += F;
            }

            if (U > s_hi) {
                double u = (s_hi - L) * inv_ds;
                int j = (int)u;
                if (j < 0) j = 0;
                if (j > Ns_local - 2) j = Ns_local - 2;

                const double t = u - (double)j;
                const double F = dvec[j] * (1.0 - t) + dvec[j+1] * t;

                tau_seg += tauU - F;
            }

            if (tau_seg > 0.0)
                sum_abs += 1.0 - exp(-tau_seg);
        }

        out[i] = sum_abs * dphi;
    }
}

/* ============================================================
   Vectorized turbulent orientation dispatch (ONE call per geometry)
   ============================================================ */
static inline void compute_phi_turbulent(
    const double *y,
    const double *yG2,
    const double *yG5,
    int n,
    double x,
    double con,
    const double *SALT,     /* alpha, psi, ... */
    const double *DIST,     /* DIST[0..4] */
    const double *GAMMA,    /* Gamma11, Gamma12 live here too */
    const double *LINE,     /* a1, ..., lambda_lu, SW, profile_method */
    double *phi_out,
    const TurbulentGrid *grid
)
{
    const double alpha = SALT[0];
    const double psi   = SALT[1];

    const double a1        = LINE[0];
    const double nu_lu     = LINE[2];
    const double Del_nu    = LINE[3];
    const double c         = LINE[4];
    const double y_c       = LINE[5];
    const double lambda_lu = LINE[7];
    const int profile_method = (LINE[9] == (double)VOIGT_COLT)
                             ? VOIGT_COLT : VOIGT_WOFZ;

    const double G11 = GAMMA[10];
    const double G12 = GAMMA[11];

    if (alpha >= M_PI/2.0) {
        Orientation_I(
            y, yG2, yG5, n,
            x, con, a1, nu_lu, Del_nu, c, y_c, lambda_lu,
            DIST[0], DIST[1],
            G11, G12, profile_method,
            Ns, phi_out, grid
        );
    } else if (alpha < psi) {
        Orientation_II(
            y, yG2, yG5, n,
            x, con, a1, nu_lu, Del_nu, c, y_c, lambda_lu,
            DIST[0], DIST[1], DIST[2], DIST[3], DIST[4],
            G11, G12, profile_method,
            Ns, Nphi, phi_out, grid
        );
    } else {
        Orientation_III(
            y, yG2, yG5, n,
            x, con, a1, nu_lu, Del_nu, c, y_c, lambda_lu,
            DIST[0], DIST[1], DIST[2], DIST[3], DIST[4],
            G11, G12, profile_method,
            Ns, Nphi, phi_out, grid
        );
    }
}

static void get_turb_Absorption_Profile(
    const double *h,
    int n,
    double x,
    const double *SALT,
    const double *DIST,
    const double *GAMMA,
    const double *LINE,
    double *s1_out,
    const TurbulentGrid *grid
)
{
    double hG2[Ny];
    double hG5[Ny];
    double phi[Ny];

    const double f_holes = SALT[5];
    const double con = SALT[8] * LINE[1];

    for (int i = 0; i < n; i++) {
        hG2[i] = h[i] * h[i];
        hG5[i] = 0.0;
        s1_out[i] = 0.0;
    }

    compute_phi_turbulent(
        h, hG2, hG5, n,
        x, con,
        SALT, DIST, GAMMA, LINE,
        phi, grid
    );

    for (int i = 0; i < n; i++)
        s1_out[i] = f_holes * h[i] * phi[i] / M_PI;
}

/* ============================================================
   IMPORTANT: Fix signatures of GI/GII/GIII/GIV
   All yG* must be POINTERS (const double*)
   ============================================================ */

/* ===================== GIV ===================== */
static void get_Absorption_Profile_GIV(
    const double *y,
    const double *yG1, const double *yG2, const double *yG3,
    const double *yG4, const double *yG5, const double *yG8, const double *yG9,
    int n,
    double absx, double x,
    const double *SALT,
    const double *GEOMETRY,
    const double *DIST,
    const double *GAMMA,
    const double *LINE,
    double *s1_out
)
{
    (void)yG2;
    (void)yG5;
    (void)x;
    (void)DIST;
    (void)LINE;

    const double Gamma2 = GAMMA[1];
    const double Gamma6 = GAMMA[5];
    const double Gamma7 = GAMMA[6];

    const double A = GEOMETRY[0];
    const double G = GEOMETRY[6];
    const double H = GEOMETRY[7];
    const double O = GEOMETRY[14];
    const double P = GEOMETRY[15];
    const double R = GEOMETRY[17];

    const double tau_0   = SALT[3];
    const double f_holes = SALT[5];

    for (int i = 0; i < n; i++) {
        const double yi = y[i];
        const double x_over_y = absx / yi;

        const double tau_y =
            (tau_0 * yG4[i]) /
            (1.0 + Gamma7 * x_over_y * x_over_y);

        s1_out[i] = 0.0;

        const double lhs = absx * yG1[i];

        if (lhs > yG3[i] * P &&
            lhs < yG3[i] * R) {

            const double p = (lhs - yG3[i] * P) / H;

            double u2 =
                (yG3[i] * A) * (yG3[i] * A)
              - (yG3[i] * A - p) * (yG3[i] * A - p);

            if (u2 < 0.0) u2 = 0.0;

            const double u = sqrt(u2);
            const double d = yG3[i] * O - p * G;
            const double f_c = Gamma6 * atan2(u, d);

            s1_out[i] =
                f_holes * f_c * Gamma2 *
                (yG8[i] + Gamma7 * absx*absx * yG9[i]) *
                (1.0 - exp(-tau_y));
        }
    }
}

/* ===================== GIII ===================== */
static void get_Absorption_Profile_GIII(
    const double *y,
    const double *yG1, const double *yG2, const double *yG3,
    const double *yG4, const double *yG5, const double *yG8, const double *yG9,
    int n,
    double absx, double x,
    const double *SALT,
    const double *GEOMETRY,
    const double *DIST,
    const double *GAMMA,
    const double *LINE,
    double *s1_out
)
{
    (void)yG2;
    (void)yG5;
    (void)x;
    (void)DIST;
    (void)LINE;

    const double Gamma2 = GAMMA[1];
    const double Gamma6 = GAMMA[5];
    const double Gamma7 = GAMMA[6];

    const double A = GEOMETRY[0];
    const double D = GEOMETRY[3];
    const double E = GEOMETRY[4];
    const double H = GEOMETRY[7];
    const double M = GEOMETRY[12];
    const double N = GEOMETRY[13];

    const double tau_0   = SALT[3];
    const double f_holes = SALT[5];

    for (int i = 0; i < n; i++) {
        const double yi = y[i];
        const double x_over_y = absx / yi;

        const double tau_y =
            (tau_0 * yG4[i]) /
            (1.0 + Gamma7 * x_over_y * x_over_y);

        s1_out[i] = 0.0;

        const double lhs = absx * yG1[i];

        if (lhs >= yG3[i] * D && lhs < yG3[i] * N) {
            const double DD = yG3[i] * M - yG3[i] * D / E - lhs / E;
            const double k  = (yG3[i] * D + lhs) / H;

            const double t = yG3[i] * A;
            double p2 = t*t - (t - k)*(t - k);
            if (p2 < 0.0) p2 = 0.0;

            const double f_c = Gamma6 * atan2(sqrt(p2), DD);

            s1_out[i] =
                f_holes * f_c * Gamma2 *
                (yG8[i] + Gamma7 * absx*absx * yG9[i]) *
                (1.0 - exp(-tau_y));
        }

        else if (lhs < yG3[i] * D) {
            const double t = yG3[i] * A;

            double DD = yG3[i] * M - yG3[i] * D / E - lhs / E;
            double k  = (yG3[i] * D + lhs) / H;

            double p2 = t*t - (t - k)*(t - k);
            if (p2 < 0.0) p2 = 0.0;

            const double f_c_u = Gamma6 * atan2(sqrt(p2), DD);

            k = (yG3[i] * D - lhs) / H;
            p2 = t*t - (t - k)*(t - k);
            if (p2 < 0.0) p2 = 0.0;

            DD = yG3[i] * M - yG3[i] * D / E + lhs / E;

            const double f_c_l = Gamma6 * atan2(sqrt(p2), DD);
            const double f_c = f_c_l + f_c_u;

            s1_out[i] =
                f_holes * f_c * Gamma2 *
                (yG8[i] + Gamma7 * absx*absx * yG9[i]) *
                (1.0 - exp(-tau_y));
        }
    }
}

/* ===================== GII ===================== */
static void get_Absorption_Profile_GII(
    const double *y,
    const double *yG1, const double *yG2, const double *yG3,
    const double *yG4, const double *yG5, const double *yG8, const double *yG9,
    int n,
    double absx, double x,
    const double *SALT,
    const double *GEOMETRY,
    const double *DIST,
    const double *GAMMA,
    const double *LINE,
    double *s1_out
)
{
    (void)DIST;
    (void)LINE;
    (void)x;

    const double Gamma2 = GAMMA[1];
    const double Gamma6 = GAMMA[5];
    const double Gamma7 = GAMMA[6];

    const double A  = GEOMETRY[0];
    const double E  = GEOMETRY[4];
    const double F  = GEOMETRY[5];
    const double H  = GEOMETRY[7];
    const double Ig = GEOMETRY[8];
    const double K  = GEOMETRY[10];
    const double O  = GEOMETRY[14];
    const double P  = GEOMETRY[15];
    const double R  = GEOMETRY[17];

    const double tau_0   = SALT[3];
    const double f_holes = SALT[5];

    for (int i = 0; i < n; i++) {
        const double yi = y[i];
        const double x_over_y = absx / yi;

        const double tau_y =
            (tau_0 * yG4[i]) /
            (1.0 + Gamma7 * x_over_y * x_over_y);

        s1_out[i] = 0.0;

        const double lhs = absx * yG1[i];

        if (lhs >= yG3[i] * R) {
            s1_out[i] =
                f_holes * Gamma2 *
                (yG8[i] + Gamma7 * absx*absx * yG9[i]) *
                (1.0 - exp(-tau_y));
        }

        else if (lhs > yG3[i] * Ig + yG3[i] * K * E &&
                 lhs < yG3[i] * R) {

            const double nval = yG3[i] * Ig + yG3[i] * O * E;
            const double v    = (lhs - nval) * F;

            const double denom = yG2[i] - absx*absx * yG5[i];

            if (denom > 0.0) {
                double arg = v / sqrt(denom);
                if (arg >  1.0) arg =  1.0;
                if (arg < -1.0) arg = -1.0;

                const double f_c = 1.0 - acos(arg) * Gamma6;

                s1_out[i] =
                    f_holes * f_c * Gamma2 *
                    (yG8[i] + Gamma7 * absx*absx * yG9[i]) *
                    (1.0 - exp(-tau_y));
            }
        }

        else if (lhs > yG3[i] * P &&
                 lhs < yG3[i] * Ig + yG3[i] * O * E) {

            const double b = (lhs - yG3[i] * Ig) / H;

            const double t = yG3[i] * A;
            double d2 = t*t - (t - b)*(t - b);
            if (d2 < 0.0) d2 = 0.0;

            const double d = sqrt(d2);
            const double h = yG3[i] * O - (lhs - yG3[i] * Ig) / E;

            const double f_c = Gamma6 * atan2(d, h);

            s1_out[i] =
                f_holes * f_c * Gamma2 *
                (yG8[i] + Gamma7 * absx*absx * yG9[i]) *
                (1.0 - exp(-tau_y));
        }
    }
}

/* ===================== GI ===================== */
static void get_Absorption_Profile_GI(
    const double *y,
    const double *yG1, const double *yG2, const double *yG3,
    const double *yG4, const double *yG5, const double *yG8, const double *yG9,
    int n,
    double absx, double x,
    const double *SALT,
    const double *GEOMETRY,
    const double *DIST,
    const double *GAMMA,
    const double *LINE,
    double *s1_out
)
{
    (void)x;
    (void)DIST;
    (void)LINE;

    const double Gamma2 = GAMMA[1];
    const double Gamma6 = GAMMA[5];
    const double Gamma7 = GAMMA[6];

    const double A = GEOMETRY[0];
    const double C = GEOMETRY[2];
    const double D = GEOMETRY[3];
    const double E = GEOMETRY[4];
    const double H = GEOMETRY[7];
    const double M = GEOMETRY[12];
    const double N = GEOMETRY[13];
    const double Q = GEOMETRY[16];

    const double tau_0   = SALT[3];
    const double f_holes = SALT[5];

    for (int i = 0; i < n; i++) {
        const double yi = y[i];
        const double x_over_y = absx / yi;

        const double tau_y =
            (tau_0 * yG4[i]) /
            (1.0 + Gamma7 * x_over_y * x_over_y);

        s1_out[i] = 0.0;

        const double lhs = absx * yG1[i];

        if (lhs > yG3[i] * N) {
            s1_out[i] =
                f_holes * Gamma2 *
                (yG8[i] + Gamma7 * absx*absx * yG9[i]) *
                (1.0 - exp(-tau_y));
            continue;
        }

        const double denom = yG2[i] - absx*absx * yG5[i];
        const double w = yG3[i] * M - yG3[i] * D / C;

        if (lhs > yG3[i] * D && lhs > C * w && lhs < yG3[i] * N) {
            if (denom > 0.0) {
                const double v = (lhs - w * C) * Q;

                double arg = v / sqrt(denom);
                if (arg >  1.0) arg =  1.0;
                if (arg < -1.0) arg = -1.0;

                const double f_c = 1.0 - acos(arg) * Gamma6;

                s1_out[i] =
                    f_holes * f_c * Gamma2 *
                    (yG8[i] + Gamma7 * absx*absx * yG9[i]) *
                    (1.0 - exp(-tau_y));
            }
            continue;
        }

        if (lhs > yG3[i] * D && lhs < C * w && lhs < yG3[i] * N) {
            const double DD = yG3[i] * M - yG3[i] * D / E - lhs / E;
            const double k  = (yG3[i] * D + lhs) / H;

            const double t = yG3[i] * A;
            double p2 = t*t - (t - k)*(t - k);
            if (p2 < 0.0) p2 = 0.0;

            const double f_c = Gamma6 * atan2(sqrt(p2), DD);

            s1_out[i] =
                f_holes * f_c * Gamma2 *
                (yG8[i] + Gamma7 * absx*absx * yG9[i]) *
                (1.0 - exp(-tau_y));
            continue;
        }

        if (lhs < yG3[i] * D && lhs >= C * w && lhs < yG3[i] * N) {
            const double t = yG3[i] * A;

            const double k_l = (yG3[i] * D - lhs) / H;
            double p2 = t*t - (t - k_l)*(t - k_l);
            if (p2 < 0.0) p2 = 0.0;

            const double DD_l = yG3[i] * M - yG3[i] * D / E + lhs / E;
            const double f_c_l = Gamma6 * atan2(sqrt(p2), DD_l);

            double f_c_u = 0.0;
            if (denom > 0.0) {
                const double v = (lhs - w * C) * Q;

                double arg = v / sqrt(denom);
                if (arg >  1.0) arg =  1.0;
                if (arg < -1.0) arg = -1.0;

                f_c_u = 1.0 - acos(arg) * Gamma6;
            }

            const double f_c = f_c_u + f_c_l;

            s1_out[i] =
                f_holes * f_c * Gamma2 *
                (yG8[i] + Gamma7 * absx*absx * yG9[i]) *
                (1.0 - exp(-tau_y));
            continue;
        }

        if (lhs < yG3[i] * D && lhs < C * w) {
            const double t = yG3[i] * A;

            const double k_l = (yG3[i] * D - lhs) / H;
            double p2 = t*t - (t - k_l)*(t - k_l);
            if (p2 < 0.0) p2 = 0.0;

            const double DD_l = yG3[i] * M - yG3[i] * D / E + lhs / E;
            const double f_c_l = Gamma6 * atan2(sqrt(p2), DD_l);

            const double k_u = (yG3[i] * D + lhs) / H;
            p2 = t*t - (t - k_u)*(t - k_u);
            if (p2 < 0.0) p2 = 0.0;

            const double DD_u = yG3[i] * M - yG3[i] * D / E - lhs / E;
            const double f_c_u = Gamma6 * atan2(sqrt(p2), DD_u);

            const double f_c = f_c_u + f_c_l;

            s1_out[i] =
                f_holes * f_c * Gamma2 *
                (yG8[i] + Gamma7 * absx*absx * yG9[i]) *
                (1.0 - exp(-tau_y));
            continue;
        }
    }
}

/*
   Computes the SALT absorption contribution at dimensionless velocity x.
   In hybrid mode, the turbulent core is evaluated for |x| < SW/|v_0|
   and the Sobolev geometry is used outside that interval.  When the Sobolev
   branch is disabled, the turbulent Voigt calculation is used everywhere
   and SW is ignored.
*/

static double Absorption_Integral(
    double x, double alpha, double psi, double y_inf,
    const double *SALT,
    const double *GAMMA,
    const double *GEOMETRY,
    int APERTURE,
    int use_sobolev_wings,
    const double *LINE,
    const double *DIST,
    const TurbulentGrid *turb_grid
)
{
    const double EPS_Y = 1e-12;
    double absx = fabs(x);
    absx = positive_floor(absx, EPS_Y);

    double scale, lower, upper;

    /* ======================================================
       turbulent regime: all x, or |x| < SW/|v_0| in hybrid mode
       ====================================================== */
    if ((!use_sobolev_wings || absx < LINE[8]) && LINE[6] > 0.0) {

        if (APERTURE && SALT[7] < 1.0) {
            scale = pow(SALT[7], GAMMA[9]);
            lower = 0.0;
            upper = pow(SALT[7], GAMMA[2]);
        } else {
            scale = 1.0;
            lower = 0.0;
            upper = 1.0;
        }

        if (upper <= lower || alpha == 0.0)
            return 0.0;

        double h[Ny];
        double f[Ny];

        const double dh = (upper - lower) / (Ny - 1);

        for (int i = 0; i < Ny; i++)
            h[i] = lower + i * dh;

        h[0] += 1e-6;

	get_turb_Absorption_Profile(
	    h, Ny,
	    x,
	    SALT, DIST, GAMMA, LINE,
	    f, turb_grid
	);

        return scale * simpson_samples(f, Ny, dh);
    } else {

      if (APERTURE && SALT[7] < 1.0) {
        scale = pow(SALT[7], GAMMA[9]);
        lower = fmax(absx, 1.0);

        if (x < 0.0) return 0.0;

        Y1APParams P = {absx, SALT[7], GAMMA[1], GAMMA[4]};

        double test =
	  pow(y_inf, GAMMA[4]) * x*x
          + pow(SALT[7], GAMMA[1])
          - pow(y_inf, GAMMA[1]);

	double rtmp;
	upper = y_inf;

	if (test < 0.0 &&
	    brent_root(y1ap_root_f, &P, lower, y_inf, &rtmp)) {
	  upper = rtmp;
	}

      } else {
        scale = 1.0;
        lower = fmax(absx, 1.0);

        if (x < 0.0) return 0.0;

        Y1Params P = {absx, GAMMA[9]};

        double test =
	  y_inf*y_inf * (1.0 - pow(y_inf, GAMMA[9])) - x*x;

	double rtmp;
	upper = y_inf;

	if (test > 0.0 &&
	    brent_root(y1_root_f, &P, lower, y_inf, &rtmp)) {
	  upper = rtmp;
	}
      }
    }

    if (upper <= lower || alpha == 0.0)
        return 0.0;

    int geometry_case;
    if (alpha + psi > M_PI/2.0 && psi - alpha <= 0.0) geometry_case = 1;
    else if (alpha + psi <= M_PI/2.0 && psi - alpha <= 0.0) geometry_case = 2;
    else if (alpha + psi > M_PI/2.0 && psi - alpha > 0.0) geometry_case = 3;
    else geometry_case = 4;

    /* This is the Sobolev branch.Do not apply these velocity-space cuts
       to the turbulent Voigt branch. */
    if ((geometry_case == 2 || geometry_case == 4) && GEOMETRY[15] > 0.0) {
        upper = fmin(upper, absx / GEOMETRY[15]);
    } else if (geometry_case == 3 && GEOMETRY[13] > 0.0) {
        lower = fmax(lower, absx / GEOMETRY[13]);
    }

    if (upper <= lower) return 0.0;

    /* ---------------- Integration grid ---------------- */
    double y[NySobolev];
    double f[NySobolev];

    const double dy = (upper - lower) / (NySobolev - 1);

    for (int i = 0; i < NySobolev; i++)
        y[i] = lower + i * dy;
    y[0] += 1e-6;

    double yG2[NySobolev];
    double yG5[NySobolev];
    double yG8[NySobolev];
    double yG9[NySobolev];
    double yG3[NySobolev];
    double yG1[NySobolev];
    double yG4[NySobolev];

    for (int i = 0; i < NySobolev; i++) {
      double yi = y[i];
      yG1[i] = pow(yi, GAMMA[0]);
      yG2[i] = pow(yi, GAMMA[1]);
      yG3[i] = pow(yi, GAMMA[2]);
      yG4[i] = pow(yi, GAMMA[3]);
      yG5[i] = pow(yi, GAMMA[4]);
      yG8[i] = pow(yi, GAMMA[7]);
      yG9[i] = pow(yi, GAMMA[8]);
    }

    /* ---------------- Geometry dispatch ---------------- */
    if (geometry_case == 1) {
      get_Absorption_Profile_GI(y, yG1, yG2, yG3, yG4, yG5, yG8, yG9, NySobolev, absx, x,
            SALT, GEOMETRY, DIST, GAMMA, LINE, f);
    }
    else if (geometry_case == 2) {
        get_Absorption_Profile_GII(y, yG1, yG2, yG3, yG4, yG5, yG8, yG9, NySobolev, absx, x,
            SALT, GEOMETRY, DIST, GAMMA, LINE, f);
    }
    else if (geometry_case == 3) {
        get_Absorption_Profile_GIII(y, yG1, yG2, yG3, yG4, yG5, yG8, yG9, NySobolev, absx, x,
            SALT, GEOMETRY, DIST, GAMMA, LINE, f);
    }
    else if (geometry_case == 4) {
        get_Absorption_Profile_GIV(y, yG1, yG2, yG3, yG4, yG5, yG8, yG9, NySobolev, absx, x,
            SALT, GEOMETRY, DIST, GAMMA, LINE, f);
    }
    else {
        return 0.0;
    }

    double Intensity = scale * simpson_samples(f, NySobolev, dy);

    return Intensity;
}

void computeABS_vector(
    double wavelength,
    double oscillator_strength,
    double einstein_coefficient,
    const double *v_obs, int n,
    double alpha, double psi, double gamma, double tau,
    double v_0, double v_w,
    double v_ap, double v_b,
    double f_holes, double delta,
    int APERTURE,
    double SW,
    int use_sobolev_wings,
    int profile_method,
    double *I_out
)
{
    /* ---------- psi == alpha fix ---------- */
    if (fabs(alpha - psi) < 1e-14 && alpha < M_PI/2.0) {
    psi = fmin(psi + 1e-12, nextafter(M_PI/2.0, 0.0));
    }

    /* ---------- Line quantities ---------- */
    const double tau_0 = wavelength * oscillator_strength * tau;
    const double A_ul  = einstein_coefficient;

    if (n <= 0) return;
    if (alpha == 0.0 || tau_0 == 0.0 || f_holes == 0.0 || v_0 == 0.0 || v_ap == 0.0) {
        for (int i = 0; i < n; ++i) I_out[i] = 0.0;
        return;
    }

    /* ---------- Geometry ---------- */
    const double A  = sin(alpha);
    const double C = tan(alpha - fabs(psi - alpha));
    const double D  = sin(psi + alpha - M_PI/2.0);
    const double E  = tan(psi);
    const double F  = tan(M_PI/2.0 - psi);
    const double G  = cos(psi);
    const double H  = sin(psi);
    const double Ig  = cos(psi + alpha);
    const double K  = sin(psi + alpha);
    const double M  = cos(psi + alpha - M_PI/2.0);
    const double N  = cos(fabs(psi - alpha));
    const double O  = sin(alpha + psi);
    const double P  = cos(alpha + psi);
    const double Q  = tan(M_PI/2.0 - alpha + fabs(psi - alpha));
    const double R  = cos(psi - alpha);

    /* ---------- Velocity scaling ---------- */
    const double y_inf = v_w / v_0;
    const double y_ap  = v_ap / v_0;
    const double y_b  = fabs(v_b / v_0);
    const double sw_normalized = use_sobolev_wings ? fabs(SW / v_0) : 0.0;
    const int turbulent_mode = (y_b > 0.0);

    if (!use_sobolev_wings && !turbulent_mode) {
        for (int i = 0; i < n; ++i) I_out[i] = 0.0;
        return;
    }

    /* ---------- DIST ---------- */
    double DIST[5];
    if (alpha >= M_PI/2.0) {
        DIST[0] = pow(y_inf, 2.0/gamma);
        DIST[1] = 2.0 * M_PI;
    } else {
        DIST[0] = pow(y_inf, 2.0/gamma);
        DIST[1] = cos(psi)*cos(psi) - cos(alpha)*cos(alpha);
        DIST[2] = 2.0 * cos(psi) * sin(psi);
        DIST[3] = sin(psi)*sin(psi) - cos(alpha)*cos(alpha);
        DIST[4] = cos(alpha)*cos(alpha);
    }

    /* ---------- GAMMA ---------- */
    const double GAMMA[14] = {
        (1.0 - gamma)/gamma,
        2.0/gamma,
        1.0/gamma,
        (1.0 - (delta + gamma))/gamma,
        2.0*(1.0 - gamma)/gamma,
        1.0/M_PI,
        gamma - 1.0,
        (2.0 - gamma)/gamma,
        (2.0 - 3.0*gamma)/gamma,
        -2.0/gamma,
        -delta/2.0,
        (gamma - 1.0)/2.0,
        gamma/delta,
        1.0/(gamma*M_PI)
    };

    /* ---------- Physical constants ---------- */
    const double c = 2.99792458e10;
    const double y_c = c / (fabs(v_0) * 1e5);
    const double nu_lu  = c / (wavelength * 1e-8);
    const double Del_nu = ((fabs(v_b) * 1e5) / c) * nu_lu;

    double a1 = 0.0, CS_0 = 0.0;
    if (Del_nu > 0.0) {
        a1   = A_ul / (4.0 * M_PI * Del_nu);
        CS_0 = 0.0265397 / sqrt(M_PI) * oscillator_strength / Del_nu;
    }

    const double n_0 =
        3.086e21 * tau * pow(0.0265397, -1.0)
        * fabs(v_0 * 1e5) / 3.086e21 * 1e8;

    /* ---------- Pack arrays ---------- */
    const double GEOMETRY[19] = {
        A, 0.0, C, D, E, F, G, H, Ig, 0.0,
        K, 0.0, M, N, O, P, Q, R, 0.0
    };

    const double SALT[9] = {
        alpha, psi, gamma, tau_0, y_inf, f_holes, delta, y_ap, n_0
    };

    const double LINE[10] = {
        a1, CS_0, nu_lu, Del_nu, c, y_c, y_b,
        wavelength * 1e-8, sw_normalized,
        (profile_method == VOIGT_COLT) ? (double)VOIGT_COLT : (double)VOIGT_WOFZ
    };

    TurbulentGrid *turb_grid = NULL;
    if (turbulent_mode && alpha != 0.0 && y_ap != 0.0) {
        const double h_upper = (APERTURE && SALT[7] < 1.0)
                             ? pow(SALT[7], GAMMA[2]) : 1.0;
        turb_grid = (TurbulentGrid*)malloc(sizeof(*turb_grid));
        if (turb_grid)
            build_turbulent_grid(turb_grid, 0.0, h_upper, DIST[0],
                                 GAMMA[10], GAMMA[11]);
    }

    /* ---------- Main loop (abs_worker parity) ---------- */
    const double inv_v0 = 1.0 / v_0;

    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int i = 0; i < n; i++) {

        double x = v_obs[i] * inv_v0;

        if (fabs(x) < 1e-12)
            x = 1e-10;

        if (turbulent_mode) {
            if (alpha == 0.0 || y_ap == 0.0)
                I_out[i] = 0.0;
            else
                I_out[i] = -Absorption_Integral(
                    x, alpha, psi, y_inf,
                    SALT, GAMMA, GEOMETRY,
                    APERTURE, use_sobolev_wings, LINE, DIST, turb_grid
                );
        } else {
            if (x < 0.0 || alpha == 0.0 || y_ap == 0.0 || fabs(x) > fabs(y_inf))
                I_out[i] = 0.0;
            else
                I_out[i] = -Absorption_Integral(
                    x, alpha, psi, y_inf,
                    SALT, GAMMA, GEOMETRY,
                    APERTURE, use_sobolev_wings, LINE, DIST, turb_grid
                );
        }
    }

    free(turb_grid);
}
