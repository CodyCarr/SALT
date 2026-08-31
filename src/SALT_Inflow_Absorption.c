/* SPDX-License-Identifier: BSD-3-Clause */

/*
 * SALT inflow absorption.
 *
 * The inflow formalism follows Carr & Scarlata (2022), ApJ, 939, 47,
 * doi:10.3847/1538-4357/ac93fa.
 *
 * This module evaluates the continuum removed by a monotonic biconical
 * inflow.  Velocities are converted to the dimensionless SALT variables
 * x=-v_obs/v_0 and y=v/v_0.  The four analytic geometry functions return the
 * fraction of an isovelocity ring that lies inside the bicone.  Fixed
 * Gauss-Legendre quadrature is split at every known geometry boundary so that
 * piecewise changes do not produce numerical ripples in the spectrum.
 *
 * The public computeABS_Inflow routine writes the absorption contribution,
 * not the transmitted flux: values lie in [-1, 0] and are combined with the
 * continuum by the line-profile assembler.
 */

#include <math.h>
#include <stddef.h>
#include <gsl/gsl_integration.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

enum { GAUSS_POINTS = 64 };

/* Avoid the relatively expensive general pow() call for common exponents. */
static inline double pow_common(double base, double exponent)
{
    if (exponent == 0.0) return 1.0;
    if (exponent == 1.0) return base;
    if (exponent == 2.0) return base * base;
    return pow(base, exponent);
}

/* Constant-impact-parameter equation used to find the second ray/shell
   intersection.  A root satisfies h(y,x)=h. */
static inline double root_equation(
    double y,
    double x,
    double Gamma1,
    double y_inf,
    double h
)
{
    const double radial = pow_common(y_inf - y, Gamma1);
    const double x_over_y = x / y;
    return radial * (1.0 - x_over_y * x_over_y) - h * h;
}

static inline double root_derivative(
    double y,
    double x,
    double Gamma1,
    double y_inf
)
{
    const double base = y_inf - y;
    const double x2 = x * x;
    const double inv_y = 1.0 / y;
    const double inv_y2 = inv_y * inv_y;
    const double radial = pow_common(base, Gamma1);
    return -Gamma1 * radial * (1.0 - x2 * inv_y2) / base
         + 2.0 * radial * x2 * inv_y2 * inv_y;
}

/* Safeguarded Newton iteration.  Bisection is used whenever the Newton step
   leaves the bracket, retaining robustness near the maximum shell height. */
static double solve_root(
    double lower,
    double upper,
    double x,
    double Gamma1,
    double y_inf,
    double h
)
{
    double f_lower = root_equation(lower, x, Gamma1, y_inf, h);
    const double f_upper_initial = root_equation(upper, x, Gamma1, y_inf, h);

    if (fabs(f_lower) < 1.0e-12) return lower;
    if (fabs(f_upper_initial) < 1.0e-12) return upper;
    if (f_lower * f_upper_initial > 0.0) return NAN;

    double current = 0.5 * (lower + upper);
    for (int iteration = 0; iteration < 64; ++iteration) {
        const double f_current = root_equation(current, x, Gamma1, y_inf, h);

        if (fabs(f_current) < 1.0e-12) return current;

        if (f_lower * f_current <= 0.0) {
            upper = current;
        } else {
            lower = current;
            f_lower = f_current;
        }

        if (fabs(upper - lower) < 1.0e-12 * fmax(1.0, fabs(current))) break;

        const double derivative = root_derivative(current, x, Gamma1, y_inf);
        const double newton = current - f_current / derivative;
        current = (isfinite(newton) && newton > lower && newton < upper)
                ? newton : 0.5 * (lower + upper);
    }

    return 0.5 * (lower + upper);
}

typedef struct {
    double gamma, tau_0, y_inf, f_c, r_ap, aperture_scale;
    double A, C, D, E, F, G, H, I, M, N, O, P, Q, R;
    double Gamma0, Gamma1, Gamma2, Gamma3, Gamma4, Gamma5;
    gsl_integration_glfixed_table *quadrature;
    int geometry, aperture;
} AbsorptionContext;

/* AbsorptionContext stores quantities that are constant across v_obs.  The
   capital letters and Gamma terms are algebraic combinations of the opening
   angle, orientation, and radial power-law indices used in the SALT
   derivation; precomputing them keeps the quadrature loop small. */

static inline double shell_height(
    const AbsorptionContext *ctx,
    double x,
    double y
);

static inline double clamp_unit(double value)
{
    if (value < -1.0) return -1.0;
    if (value >  1.0) return  1.0;
    return value;
}

static inline double safe_sqrt(double value)
{
    return sqrt(fmax(0.0, value));
}

/* Geometry I: the line of sight lies within the bicone and the two cones
   overlap in projection (alpha + psi > pi/2 and psi <= alpha). */
static inline double GeometryI_scalar(
    double y,
    double x,
    double y_inf,
    double A,
    double C,
    double D,
    double E,
    double H,
    double M,
    double N,
    double Q,
    double Gamma1,
    double Gamma2,
    double Gamma4
)
{
    const double base = y_inf - y;
    const double r2 = pow_common(base, Gamma2);
    const double r1 = r2 * r2;
    const double x_over_y = x / y;
    const double projected = r2 * x_over_y;
    const double boundary_d = r2 * D;
    const double boundary_n = r2 * N;
    const double boundary_c = C * r2 * M - boundary_d;
    const double height = safe_sqrt(r1 * (1.0 - x_over_y * x_over_y));

    double w, v, ee, kk, p, denominator;
    double f_g_lower, f_g_upper;

    if (projected > boundary_n) return 1.0;

    if (
        projected > boundary_d &&
        projected > boundary_c &&
        projected < boundary_n
    ) {
        w = r2 * M - r2 * D / C;
        v = (projected - w * C) * Q;
        ee = acos(clamp_unit(v / height));
        return 1.0 - ee * Gamma4;
    }

    if (
        projected > boundary_d &&
        projected < boundary_c &&
        projected < boundary_n
    ) {
        denominator = r2 * M - r2 * D / E - projected / E;
        kk = (boundary_d + projected) / H;
        p = safe_sqrt((r2 * A) * (r2 * A) - (r2 * A - kk) * (r2 * A - kk));
        return Gamma4 * atan(p / denominator);
    }

    if (
        projected < boundary_d &&
        projected >= boundary_c &&
        projected < boundary_n
    ) {
        kk = (boundary_d - projected) / H;
        p = safe_sqrt((r2 * A) * (r2 * A) - (r2 * A - kk) * (r2 * A - kk));
        denominator = r2 * M - r2 * D / E + projected / E;
        f_g_lower = Gamma4 * atan(p / denominator);

        w = r2 * M - r2 * D / C;
        v = (projected - w * C) * Q;
        ee = acos(clamp_unit(v / height));
        f_g_upper = 1.0 - ee * Gamma4;

        return f_g_upper + f_g_lower;
    }

    if (
        projected < boundary_d &&
        projected < boundary_c
    ) {
        kk = (boundary_d - projected) / H;
        p = safe_sqrt((r2 * A) * (r2 * A) - (r2 * A - kk) * (r2 * A - kk));
        denominator = r2 * M - r2 * D / E + projected / E;
        f_g_lower = Gamma4 * atan(p / denominator);

        kk = (boundary_d + projected) / H;
        p = safe_sqrt((r2 * A) * (r2 * A) - (r2 * A - kk) * (r2 * A - kk));
        denominator = r2 * M - r2 * D / E - projected / E;
        f_g_upper = Gamma4 * atan(p / denominator);

        return f_g_upper + f_g_lower;
    }

    return 0.0;
}

/* Geometry II: the line of sight lies within the bicone without projected
   overlap (alpha + psi <= pi/2 and psi <= alpha). */
static inline double GeometryII_scalar(
    double y,
    double x,
    double y_inf,
    double A,
    double E,
    double F,
    double H,
    double I,
    double O,
    double P,
    double R,
    double Gamma1,
    double Gamma2,
    double Gamma4
)
{
    const double base = y_inf - y;
    const double r2 = pow_common(base, Gamma2);
    const double r1 = r2 * r2;
    const double x_over_y = x / y;
    const double projected = r2 * x_over_y;
    const double boundary_r = r2 * R;
    const double boundary_upper = r2 * I + r2 * O * E;
    const double boundary_p = r2 * P;
    const double height = safe_sqrt(r1 * (1.0 - x_over_y * x_over_y));

    double n, v, ee, b, d, h;

    if (projected >= boundary_r) return 1.0;

    if (
        projected > boundary_upper &&
        projected < boundary_r
    ) {
        n = r2 * I + r2 * O * E;
        v = (projected - n) * F;
        ee = acos(clamp_unit(v / height));
        return 1.0 - ee * Gamma4;
    }

    if (
        projected > boundary_p &&
        projected < r2 * I + r2 * O * E
    ) {
        b = (projected - r2 * I) / H;
        d = safe_sqrt((r2 * A) * (r2 * A) - (r2 * A - b) * (r2 * A - b));
        h = r2 * O - (projected - r2 * I) / E;
        return Gamma4 * atan(d / h);
    }

    return 0.0;
}

/* Geometry III: the line of sight lies outside the cone and the projected
   front and back cones overlap (alpha + psi > pi/2 and psi > alpha). */
static inline double GeometryIII_scalar(
    double y,
    double x,
    double y_inf,
    double A,
    double D,
    double E,
    double H,
    double M,
    double N,
    double Gamma2,
    double Gamma4
)
{
    const double r2 = pow_common(y_inf - y, Gamma2);
    const double projected = r2 * (x / y);
    const double boundary_d = r2 * D;
    const double boundary_n = r2 * N;
    const double radial_a = r2 * A;

    double denominator, k, p;
    double f_g_lower, f_g_upper;

    if (
        projected >= boundary_d &&
        projected < boundary_n
    ) {
        denominator = r2 * M - boundary_d / E - projected / E;
        k = (boundary_d + projected) / H;
        p = safe_sqrt(radial_a * radial_a - (radial_a - k) * (radial_a - k));

        return Gamma4 * atan(p / denominator);
    }

    if (projected < boundary_d) {
        denominator = r2 * M - boundary_d / E - projected / E;
        k = (boundary_d + projected) / H;
        p = safe_sqrt(radial_a * radial_a - (radial_a - k) * (radial_a - k));
        f_g_upper = Gamma4 * atan(p / denominator);

        k = (boundary_d - projected) / H;
        p = safe_sqrt(radial_a * radial_a - (radial_a - k) * (radial_a - k));
        denominator = r2 * M - boundary_d / E + projected / E;
        f_g_lower = Gamma4 * atan(p / denominator);

        return f_g_lower + f_g_upper;
    }

    return 0.0;
}

/* Geometry IV: the line of sight lies outside the cone without projected
   overlap (alpha + psi <= pi/2 and psi > alpha). */
static inline double GeometryIV_scalar(
    double y,
    double x,
    double y_inf,
    double A,
    double G,
    double H,
    double O,
    double P,
    double R,
    double Gamma2,
    double Gamma4
)
{
    const double r2 = pow_common(y_inf - y, Gamma2);
    const double projected = r2 * (x / y);
    const double boundary_p = r2 * P;
    const double boundary_r = r2 * R;

    if (
        projected > boundary_p &&
        projected < boundary_r
    ) {
        const double p = (projected - boundary_p) / H;
        const double radial_a = r2 * A;
        const double u = safe_sqrt(
            radial_a * radial_a -
            (radial_a - p) * (radial_a - p)
        );
        const double d = r2 * O - p * G;

        return Gamma4 * atan(u / d);
    }

    return 0.0;
}

/* Dispatch the analytic bicone case and apply the circular aperture in the
   plane of the sky. */
static inline double absorption_geometry(
    const AbsorptionContext *ctx,
    double y,
    double x
)
{
    double geometry;
    switch (ctx->geometry) {
        case 1:
            geometry = GeometryI_scalar(
                y, x, ctx->y_inf,
                ctx->A, ctx->C, ctx->D, ctx->E, ctx->H,
                ctx->M, ctx->N, ctx->Q,
                ctx->Gamma1, ctx->Gamma2, ctx->Gamma4
            );
            break;

        case 2:
            geometry = GeometryII_scalar(
                y, x, ctx->y_inf,
                ctx->A, ctx->E, ctx->F, ctx->H, ctx->I,
                ctx->O, ctx->P, ctx->R,
                ctx->Gamma1, ctx->Gamma2, ctx->Gamma4
            );
            break;

        case 3:
            geometry = GeometryIII_scalar(
                y, x, ctx->y_inf,
                ctx->A, ctx->D, ctx->E, ctx->H,
                ctx->M, ctx->N,
                ctx->Gamma2, ctx->Gamma4
            );
            break;

        case 4:
            geometry = GeometryIV_scalar(
                y, x, ctx->y_inf,
                ctx->A, ctx->G, ctx->H, ctx->O,
                ctx->P, ctx->R,
                ctx->Gamma2, ctx->Gamma4
            );
            break;

        default:
            return 0.0;
    }
    if (ctx->aperture && shell_height(ctx, x, y) > ctx->r_ap) return 0.0;
    return geometry;
}

/* Sobolev optical depth at a shell point, including the projected velocity
   gradient along the ray. */
static inline double optical_depth(
    const AbsorptionContext *ctx,
    double x,
    double y
)
{
    const double base = ctx->y_inf - y;
    const double x_over_y = x / y;

    const double denominator =
        y +
        (
            ctx->Gamma0 * base - y
        ) * x_over_y * x_over_y;

    return ctx->tau_0 * pow_common(base, ctx->Gamma3) / denominator;
}

/* Jacobian that converts integration over shell velocity y into the projected
   area removed from the continuum at observed velocity x. */
static inline double absorption_area(
    const AbsorptionContext *ctx,
    double x,
    double y
)
{
    const double base = ctx->y_inf - y;
    const double x2 = x * x;
    const double y2 = y * y;
    const double y3 = y2 * y;
    const double radial_1 = pow_common(base, ctx->Gamma1);
    const double radial_5 = pow_common(base, ctx->Gamma5);

    return ctx->Gamma1 * (
        (x2 / y2) * radial_5 +
        (ctx->Gamma0 * x2 / y3) * radial_1 -
        radial_5
    );
}

/* Projected impact parameter in units of the source radius. */
static inline double shell_height(
    const AbsorptionContext *ctx,
    double x,
    double y
)
{
    const double radial = pow_common(ctx->y_inf - y, ctx->Gamma1);
    const double x_over_y = x / y;

    return sqrt(fmax(
        0.0,
        radial * (1.0 - x_over_y * x_over_y)
    ));
}


enum {
    STANDARD_SEGMENT = 0,
    OVERLAP_SEGMENT = 1
};

/* Integrate one smooth interval.  OVERLAP_SEGMENT additionally attenuates the
   near-side contribution by material at the second shell intersection y_2. */
static double integrate_absorption_interval(
    const AbsorptionContext *ctx,
    double x,
    double lower,
    double upper,
    int segment_type,
    double y_max,
    double h_lower,
    double h_upper
)
{
    if (!(upper > lower)) return 0.0;

    double integral = 0.0;

    for (int index = 0; index < GAUSS_POINTS; ++index) {
        double y, weight;
        gsl_integration_glfixed_point(lower, upper, (size_t)index,
                                      &y, &weight, ctx->quadrature);

        const double f_g_1 = absorption_geometry(ctx, y, x);
        if (f_g_1 == 0.0 || ctx->f_c == 0.0) continue;
        const double tau_1 = optical_depth(ctx, x, y);
        const double area = absorption_area(ctx, x, y);
        const double absorbed_1 = -expm1(-tau_1);
        if (absorbed_1 == 0.0 || area == 0.0) continue;
        double value = ctx->f_c * f_g_1 * area * absorbed_1;

        if (segment_type == STANDARD_SEGMENT) {
            integral += weight * value;
            continue;
        }

        double h_2 = shell_height(ctx, x, y);

        if (h_2 >= h_upper - 1.0e-6) h_2 = h_upper - 1.0e-6;
        if (h_2 <= h_lower + 1.0e-6) h_2 = h_lower + 1.0e-6;

        const double y_2 = solve_root(
            y_max,
            ctx->y_inf - 1.0,
            x,
            ctx->Gamma1,
            ctx->y_inf,
            h_2
        );

        if (!isfinite(y_2)) {
            continue;
        }

        const double f_g_2 = absorption_geometry(ctx, y_2, x);
        double transmission_2 = 1.0;
        if (f_g_2 != 0.0) {
            const double tau_2 = optical_depth(ctx, x, y_2);
            transmission_2 -= f_g_2*(-expm1(-tau_2));
        }

        value *= transmission_2;
        integral += weight * value;
    }

    return integral;
}

/* The geometry factors are piecewise smooth in q=x/y.  Splitting at their
   analytic q boundaries prevents a moving discontinuity from crossing the
   fixed Gauss nodes and imprinting small oscillations on the spectrum. */
static int geometry_q_boundaries(
    const AbsorptionContext *ctx,
    double boundaries[3]
)
{
    switch (ctx->geometry) {
        case 1:
            boundaries[0] = ctx->N;
            boundaries[1] = ctx->D;
            boundaries[2] = ctx->C * ctx->M - ctx->D;
            return 3;
        case 2:
            boundaries[0] = ctx->R;
            boundaries[1] = ctx->I + ctx->O * ctx->E;
            boundaries[2] = ctx->P;
            return 3;
        case 3:
            boundaries[0] = ctx->D;
            boundaries[1] = ctx->N;
            return 2;
        case 4:
            boundaries[0] = ctx->P;
            boundaries[1] = ctx->R;
            return 2;
        default:
            return 0;
    }
}

/* Split a requested interval at geometry and aperture crossings, then sum
   fixed-order quadrature over the resulting smooth subintervals. */
static double integrate_absorption_segment(
    const AbsorptionContext *ctx,
    double x,
    double lower,
    double upper,
    int segment_type,
    double y_max,
    double h_lower,
    double h_upper
)
{
    if (!(upper > lower)) return 0.0;

    double points[12] = {lower, upper};
    int n_points = 2;
    double q_boundaries[3];
    const int n_boundaries = geometry_q_boundaries(ctx, q_boundaries);

    if (ctx->aperture && ctx->r_ap < 1.0) {
        const double outer = ctx->y_inf - 1.0;
        const double left = solve_root(
            x, y_max, x, ctx->Gamma1, ctx->y_inf, ctx->r_ap
        );
        const double right = solve_root(
            y_max, outer, x, ctx->Gamma1, ctx->y_inf, ctx->r_ap
        );
        if (isfinite(left) && left > lower && left < upper) points[n_points++] = left;
        if (isfinite(right) && right > lower && right < upper) points[n_points++] = right;
    }

    for (int i = 0; i < n_boundaries; ++i) {
        const double q = q_boundaries[i];
        if (q <= 0.0 || !isfinite(q)) continue;
        const double crossing = x / q;
        if (crossing > lower && crossing < upper) points[n_points++] = crossing;

        /* In an overlap segment the far-side intersection y_2 carries a
           second geometry factor.  Mirror its boundary onto the near-side
           integration coordinate so that discontinuity is split as well. */
        if (
            segment_type == OVERLAP_SEGMENT &&
            crossing > y_max &&
            crossing < ctx->y_inf - 1.0
        ) {
            const double crossing_height = shell_height(ctx, x, crossing);
            const double mirrored = solve_root(
                x, y_max, x, ctx->Gamma1, ctx->y_inf, crossing_height
            );
            if (isfinite(mirrored) && mirrored > lower && mirrored < upper) {
                points[n_points++] = mirrored;
            }
        }
    }

    for (int i = 1; i < n_points; ++i) {
        const double value = points[i];
        int j = i;
        while (j > 0 && points[j - 1] > value) {
            points[j] = points[j - 1];
            --j;
        }
        points[j] = value;
    }

    double integral = 0.0;
    for (int i = 0; i + 1 < n_points; ++i) {
        if (points[i + 1] - points[i] <= 1.0e-13) continue;
        integral += integrate_absorption_interval(
            ctx, x, points[i], points[i + 1], segment_type,
            y_max, h_lower, h_upper
        );
    }
    return integral;
}

/* Construct the complete absorption integral at one dimensionless observed
   velocity.  The three pieces distinguish unobscured and overlapping ray
   segments on either side of the maximum impact parameter. */
static double Absorption_Integral(
    double x,
    const AbsorptionContext *ctx
)
{
    const double x2 = x * x;
    const double q = -ctx->gamma * x2 * ctx->y_inf;
    const double p = (ctx->gamma - 1.0) * x2;
    const double discriminant = fmax(0.0, q * q / 4.0 + p * p * p / 27.0);
    const double root_discriminant = sqrt(discriminant);
    const double y_max = cbrt(-q / 2.0 + root_discriminant) + cbrt(-q / 2.0 - root_discriminant);
    const double outer_boundary = ctx->y_inf - 1.0;

    if (y_max >= outer_boundary) {
        return integrate_absorption_segment(
            ctx,
            x,
            x,
            outer_boundary,
            STANDARD_SEGMENT,
            y_max,
            0.0,
            0.0
        );
    }

    const double h_max = shell_height(ctx, x, y_max);
    const double h_s = sqrt(fmax(
        0.0,
        1.0 - (x / outer_boundary) * (x / outer_boundary)
    ));

    const double upper_1 = solve_root(
        x,
        y_max,
        x,
        ctx->Gamma1,
        ctx->y_inf,
        h_s
    );

    if (!isfinite(upper_1)) return 0.0;

    double upper_2;
    double lower_3;
    double overlap_height;

    if (h_max > 1.0) {
        upper_2 = solve_root(
            x,
            y_max,
            x,
            ctx->Gamma1,
            ctx->y_inf,
            1.0
        );

        lower_3 = solve_root(
            y_max,
            outer_boundary,
            x,
            ctx->Gamma1,
            ctx->y_inf,
            1.0
        );

        overlap_height = 1.0;
    } else {
        upper_2 = y_max;
        lower_3 = y_max;
        overlap_height = h_max;
    }

    if (!isfinite(upper_2) || !isfinite(lower_3)) return 0.0;

    const double part_1 = integrate_absorption_segment(
        ctx,
        x,
        x,
        upper_1,
        STANDARD_SEGMENT,
        y_max,
        h_s,
        overlap_height
    );

    const double part_2 = integrate_absorption_segment(
        ctx,
        x,
        upper_1,
        upper_2,
        OVERLAP_SEGMENT,
        y_max,
        h_s,
        overlap_height
    );

    const double part_3 = integrate_absorption_segment(
        ctx,
        x,
        lower_3,
        outer_boundary,
        STANDARD_SEGMENT,
        y_max,
        h_s,
        overlap_height
    );

    return part_1 + part_2 - part_3;
}

/*
 * Compute the absorption contribution for one transition over v_obs.
 *
 * wavelength is in Angstrom and all velocities are in km s^-1.  The caller
 * passes the signed internal inflow velocities (the line-profile layer
 * changes positive user magnitudes to negative values).  OpenMP is used only
 * for sufficiently large velocity arrays to avoid parallel-startup overhead.
 */
void computeABS_Inflow(
    double wavelength,
    double oscillator_strength,
    const double *v_obs,
    size_t n_velocity,
    double alpha,
    double psi,
    double gamma,
    double tau,
    double v_0,
    double v_w,
    double v_ap,
    double f_c,
    double delta,
    int aperture,
    double *absorption
)
{
    AbsorptionContext ctx = {0};

    /* Quantities below do not depend on observed velocity. */
    ctx.gamma = gamma;
    ctx.tau_0 = wavelength * oscillator_strength * tau;
    ctx.y_inf = v_w / v_0;
    ctx.f_c = f_c;
    double y_ap = v_ap / v_0;
    if (y_ap > ctx.y_inf) y_ap = ctx.y_inf;
    ctx.r_ap = pow_common(ctx.y_inf - y_ap, 1.0 / gamma);
    ctx.aperture = aperture;
    ctx.aperture_scale = aperture && ctx.r_ap < 1.0
                       ? 1.0 / (ctx.r_ap * ctx.r_ap) : 1.0;

    ctx.A = sin(alpha);
    ctx.C = tan(alpha - fabs(psi - alpha));
    ctx.D = sin(psi + alpha - M_PI / 2.0);
    ctx.E = tan(psi);
    ctx.F = tan(M_PI / 2.0 - psi);
    ctx.G = cos(psi);
    ctx.H = sin(psi);
    ctx.I = cos(psi + alpha);
    ctx.M = cos(psi + alpha - M_PI / 2.0);
    ctx.N = cos(fabs(psi - alpha));
    ctx.O = sin(alpha + psi);
    ctx.P = cos(alpha + psi);
    ctx.Q = tan(M_PI / 2.0 - alpha + fabs(psi - alpha));
    ctx.R = cos(psi - alpha);

    ctx.Gamma0 = gamma;
    ctx.Gamma1 = 2.0 / gamma;
    ctx.Gamma2 = 1.0 / gamma;
    ctx.Gamma3 = (1.0 - delta) / gamma;
    ctx.Gamma4 = 1.0 / M_PI;
    ctx.Gamma5 = (2.0 - gamma) / gamma;
    ctx.quadrature = gsl_integration_glfixed_table_alloc(GAUSS_POINTS);

    if (ctx.quadrature == NULL) {
        for (size_t index = 0; index < n_velocity; ++index) absorption[index] = NAN;
        return;
    }

    /* Select one mutually exclusive analytic projection geometry. */
    if (alpha + psi > M_PI / 2.0 && psi - alpha <= 0.0) {
        ctx.geometry = 1;
    } else if (alpha + psi <= M_PI / 2.0 && psi - alpha <= 0.0) {
        ctx.geometry = 2;
    } else if (alpha + psi > M_PI / 2.0 && psi - alpha > 0.0) {
        ctx.geometry = 3;
    } else {
        ctx.geometry = 4;
    }

    #ifdef _OPENMP
    #pragma omp parallel for schedule(static) if(n_velocity >= 512U)
    #endif
    for (size_t index = 0; index < n_velocity; ++index) {
        /* With signed v_0<0 for an inflow, absorption support maps to x>0. */
        double x = -v_obs[index] / v_0;

        if (x > 0.0 && x < 1.0e-5) x = 1.0e-5;
        if (x <= 0.0 && x > -1.0e-5) x = -1.0e-5;

        if (
            x > ctx.y_inf - 1.0 ||
            alpha == 0.0 ||
            ctx.tau_0 == 0.0 ||
            x < 0.0
        ) {
            absorption[index] = 0.0;
        } else {
            double absorbed = ctx.aperture_scale * Absorption_Integral(x, &ctx);
            if (absorbed < 0.0) absorbed = 0.0;
            if (absorbed > 1.0) absorbed = 1.0;
            absorption[index] = -absorbed;
        }
    }

    gsl_integration_glfixed_table_free(ctx.quadrature);
}
