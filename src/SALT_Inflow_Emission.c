/* SPDX-License-Identifier: BSD-3-Clause */

/*
 * SALT inflow resonant and fluorescent re-emission.
 *
 * The inflow formalism follows Carr & Scarlata (2022), ApJ, 939, 47,
 * doi:10.3847/1538-4357/ac93fa.
 *
 * The outer integral sums emission over shells and the inner integral computes
 * the continuum luminosity absorbed by an individual shell.  The calculation
 * includes biconical projection, dust attenuation, a finite aperture,
 * occultation by the source, and escape probabilities for resonant and
 * fluorescent channels.  Fixed Gauss-Legendre nodes are cached in the context
 * because these routines are evaluated for every observed velocity.
 */

#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_sf_hyperg.h>
#include <complex.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

enum {
    EMISSION_OUTER_POINTS = 128,
    EMISSION_INNER_POINTS = 64,
    SPHERICAL_OUTER_POINTS = 128,
    SPHERICAL_INNER_POINTS = 32
};

/* The spherical case is common and smooth, so it uses a smaller inner rule;
   biconical cases retain additional nodes around their geometry boundaries. */

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

static inline double pow_common(double base, double exponent)
{
    if (exponent == 0.0) return 1.0;
    if (exponent == 1.0) return base;
    if (exponent == 2.0) return base * base;
    return pow(base, exponent);
}

/* Constant-impact-parameter equation for the second ray/shell intersection. */
static inline double emission_root_equation(
    double y,
    double x,
    double Gamma2,
    double y_inf,
    double h
)
{
    const double radial = pow_common(y_inf - y, Gamma2);
    const double x_over_y = x / y;

    return radial * (1.0 - x_over_y * x_over_y) - h * h;
}

/* Bracketed bisection used where a ray intersects the same impact parameter
   on the far side of the maximum-height shell. */
static double solve_emission_root(
    double lower, double upper, double x,
    double Gamma2, double y_inf, double h
)
{
    double f_lower = emission_root_equation(lower, x, Gamma2, y_inf, h);
    double f_upper = emission_root_equation(upper, x, Gamma2, y_inf, h);

    if (fabs(f_lower) < 1.0e-12) return lower;
    if (fabs(f_upper) < 1.0e-12) return upper;
    if (!isfinite(f_lower) || !isfinite(f_upper) || f_lower * f_upper > 0.0) return NAN;

    for (int iteration = 0; iteration < 64; ++iteration) {
        const double middle = 0.5 * (lower + upper);
        const double f_middle = emission_root_equation(middle, x, Gamma2, y_inf, h);
        if (!isfinite(f_middle)) return NAN;
        if (fabs(f_middle) < 1.0e-12) return middle;
        if (f_lower * f_middle <= 0.0) { upper = middle; f_upper = f_middle; }
        else { lower = middle; f_lower = f_middle; }
        if (fabs(upper - lower) < 1.0e-12 * fmax(1.0, fabs(middle))) break;
    }

    return 0.5 * (lower + upper);
}

/* Evaluate 2F1 on the negative real axis.  Pfaff's transformation moves
   arguments below -1 into GSL's better-conditioned interval (0,1). */
static inline double hyperg_2F1_negative(
    double a,
    double b,
    double c,
    double z
)
{
    if (!isfinite(z)) return NAN;
    if (z == 0.0) return 1.0;

    if (z > -1.0) {
        return gsl_sf_hyperg_2F1(a, b, c, z);
    }

    /*
     * Pfaff transformation:
     *
     * 2F1(a,b;c;z) =
     * (1-z)^(-a) 2F1(a,c-b;c;z/(z-1)).
     *
     * For z < -1, the transformed argument lies between 0 and 1.
     */
    const double transformed_z = z / (z - 1.0);

    return pow(1.0 - z, -a) *
           gsl_sf_hyperg_2F1(
               a,
               c - b,
               c,
               transformed_z
           );
}

/* Dust optical depth from the emitting point to the wind boundary.  The
   analytic expression is evaluated along the observer's ray. */
static inline double dust_scalar(
    double y,
    double x,
    double y_inf,
    double k_dust,
    double delta,
    double Gamma3,
    double Gamma5,
    double Gamma12,
    double Gamma13,
    double Gamma14,
    double Gamma15,
    double Gamma16
)
{
    if (k_dust == 0.0) return 0.0;

    const double x_over_y = x / y;

    double cos_theta;
    double sin_theta;

    if (x_over_y <= -1.0) {
        cos_theta = -1.0;
        sin_theta = 0.0;
    } else if (x_over_y >= 1.0) {
        cos_theta = 1.0;
        sin_theta = 0.0;
    } else {
        cos_theta = x_over_y;
        sin_theta = sqrt(fmax(0.0, 1.0 - x_over_y * x_over_y));
    }

    const double y_gamma3 = pow_common(y, Gamma3);
    double z = y_gamma3 * sin_theta;

    /*
     * The Python threshold is 10e-10, which equals 1e-9.
     */
    if (z < 1.0e-9) z = 0.0;

    double tau_d;

    if (z == 0.0) {
        if (delta != 1.0) {
            tau_d = (
                k_dust *
                Gamma5 *
                (Gamma13 - pow_common(y, Gamma14))
            );
        } else {
            tau_d = (
                k_dust *
                Gamma3 *
                log(y_inf / y)
            );
        }
    } else {
        const double z2 = z * z;
        const double z_gamma16 = pow_common(z, Gamma16);

        /*
         * Gamma12 is y_inf^Gamma3, so
         *
         * Gamma12*cos(asin(z/Gamma12))
         *
         * is equivalent to sqrt(Gamma12^2-z^2).
         */
        const double first_position = sqrt(
            fmax(0.0, Gamma12 * Gamma12 - z2)
        );

        const double second_position = y_gamma3 * cos_theta;

        const double first_argument = -(
            first_position * first_position
        ) / z2;

        const double second_argument = -(
            second_position * second_position
        ) / z2;

        const double first_hyperg = hyperg_2F1_negative(
            0.5,
            Gamma15,
            1.5,
            first_argument
        );

        const double second_hyperg = hyperg_2F1_negative(
            0.5,
            Gamma15,
            1.5,
            second_argument
        );

        tau_d = k_dust * (
            first_position * first_hyperg / z_gamma16 -
            second_position * second_hyperg / z_gamma16
        );
    }

    if (!isfinite(tau_d)) return 0.0;
    return fmax(0.0, tau_d);
}

/* Third-order Bernoulli approximation to the angle-averaged escape
   probability in the optically thin/intermediate regime. */
static inline double getBernoulli3(
    double y,
    double tau,
    double y_inf,
    double Gamma4,
    double Gamma9
)
{
    const double radial = y_inf - y;
    const double a = tau * pow_common(radial, Gamma4);

    /*
     * Use the analytic limiting expression near Gamma9 = 0.
     * A tolerance is safer than an exact floating-point comparison.
     */
    if (fabs(Gamma9) < 1.0e-14) {
        const double a2 = a * a;
        return 1.0 / (1.0 + 0.5 * a + a2 / 12.0);
    }

    const double sqrt3 = sqrt(3.0);
    const double complex sqrt_gamma9 = csqrt(Gamma9 + 0.0 * _Complex_I);

    const double complex numerator_root_1 = csqrt(
        -_Complex_I * a / sqrt3 + a + 4.0
    );

    const double complex numerator_root_2 = csqrt(
        +_Complex_I * a / sqrt3 + a + 4.0
    );

    const double complex denominator_root_1 = csqrt(
        12.0 + (3.0 - sqrt3 * _Complex_I) * a
    );

    const double complex denominator_root_2 = csqrt(
        12.0 + (3.0 + sqrt3 * _Complex_I) * a
    );

    const double complex term_1 =
        (sqrt3 + _Complex_I) *
        a *
        catan(2.0 * sqrt_gamma9 / numerator_root_1) /
        (sqrt_gamma9 * denominator_root_1);

    const double complex term_2 =
        (sqrt3 - _Complex_I) *
        a *
        catan(2.0 * sqrt_gamma9 / numerator_root_2) /
        (sqrt_gamma9 * denominator_root_2);

    const double result = creal(
        0.5 * (2.0 - term_1 - term_2)
    );

    return isfinite(result) ? result : 0.0;
}

/* Optically thick asymptotic escape probability. */
static inline double getAsymptotic(
    double y,
    double tau,
    double y_inf,
    double Gamma4,
    double Gamma9
)
{
    const double radial = y_inf - y;

    return (
        (3.0 + Gamma9) /
        (
            3.0 *
            tau *
            pow_common(radial, Gamma4)
        )
    );
}

/* Select the thin/intermediate or thick approximation, optionally switching
   at a precomputed shell coordinate where the two regimes meet. */
static inline double getBeta(
    double y,
    double tau,
    double y_inf,
    double Gamma4,
    double Gamma9,
    double Gamma17,
    double change_approx
)
{
    if (change_approx == 0.0) {
        if (Gamma17 > 1.0) {
            return getAsymptotic(
                y,
                tau,
                y_inf,
                Gamma4,
                Gamma9
            );
        }

        return getBernoulli3(
            y,
            tau,
            y_inf,
            Gamma4,
            Gamma9
        );
    }

    if (y < change_approx) {
        return getAsymptotic(
            y,
            tau,
            y_inf,
            Gamma4,
            Gamma9
        );
    }

    return getBernoulli3(
        y,
        tau,
        y_inf,
        Gamma4,
        Gamma9
    );
}

typedef struct {
    double y_inf, r_ap, tau_0, f_c, k_dust, delta;

    double A, C, D, E, F, G, H;
    double Ig, M, N, O, P, Q, R;

    double Gamma0;
    double Gamma2;
    double Gamma3;
    double Gamma4;
    double Gamma5;
    double Gamma8;
    double Gamma9;
    double Gamma10;
    double Gamma12;
    double Gamma13;
    double Gamma14;
    double Gamma15;
    double Gamma16;
    double Gamma17;
    double Gamma18;
    double Gamma20;

    double p_r, p_f, change_approx;
    int resonance, fluorescence, occultation, aperture;

    int geometry, outer_points, inner_points;

    const double *flux_x;
    const double *flux;
    size_t n_flux;
    double flux_x0, flux_inv_step;
    double flux_constant_value;
    int flux_uniform, flux_constant;
    double outer_node[EMISSION_OUTER_POINTS];
    double outer_weight[EMISSION_OUTER_POINTS];
    double inner_node[EMISSION_INNER_POINTS];
    double inner_weight[EMISSION_INNER_POINTS];
} EmissionContext;

/* EmissionContext caches geometry coefficients, power-law combinations,
   interpolation metadata, and quadrature nodes shared by all velocity bins. */

static inline double emission_geometry_scalar(
    const EmissionContext *ctx,
    double y,
    double x,
    int red_side
);

/* Interpolate the incident absorbed spectrum.  Constant and uniformly spaced
   arrays use fast paths; arbitrary monotonic grids use binary search. */
static inline double interpolate_flux(
    const EmissionContext *ctx,
    double x
)
{
    if (ctx->n_flux == 0) return 0.0;
    if (ctx->n_flux == 1) return ctx->flux[0];
    if (ctx->flux_constant) return ctx->flux_constant_value;

    if (x <= ctx->flux_x[0]) return ctx->flux[0];
    if (x >= ctx->flux_x[ctx->n_flux - 1]) {
        return ctx->flux[ctx->n_flux - 1];
    }

    if (ctx->flux_uniform) {
        const double position = (x - ctx->flux_x0) * ctx->flux_inv_step;
        size_t lower = (size_t)position;
        if (lower >= ctx->n_flux - 1) lower = ctx->n_flux - 2;
        const double fraction = position - (double)lower;
        return ctx->flux[lower] + fraction * (ctx->flux[lower + 1] - ctx->flux[lower]);
    }

    size_t lower = 0;
    size_t upper = ctx->n_flux - 1;

    while (upper - lower > 1) {
        const size_t middle = lower + (upper - lower) / 2;

        if (ctx->flux_x[middle] <= x) {
            lower = middle;
        } else {
            upper = middle;
        }
    }

    const double fraction = (
        (x - ctx->flux_x[lower]) /
        (ctx->flux_x[upper] - ctx->flux_x[lower])
    );

    return (
        ctx->flux[lower] +
        fraction * (
            ctx->flux[upper] -
            ctx->flux[lower]
        )
    );
}

/* Shell coordinate at which the projected impact parameter reaches its
   maximum for a fixed observed velocity. */
static inline double emission_y_max(
    const EmissionContext *ctx,
    double x
)
{
    const double x2 = x * x;

    /* For gamma == 1, the depressed cubic has p == 0 and reduces exactly
       to y_max = cbrt(y_inf*x^2).  This is the common model configuration
       and avoids a square root plus a second cube root in the inner loop. */
    if (ctx->Gamma9 == 0.0) return cbrt(ctx->y_inf * x2);

    const double q = ctx->Gamma20 * x2;
    const double p = ctx->Gamma9 * x2;

    const double discriminant = fmax(
        0.0,
        q * q / 4.0 +
        p * p * p / 27.0
    );

    const double root_discriminant = sqrt(discriminant);

    return (
        cbrt(-q / 2.0 + root_discriminant) +
        cbrt(-q / 2.0 - root_discriminant)
    );
}

/* Differential shell-area/Jacobian factor in the emitted luminosity. */
static inline double shell_factor(
    const EmissionContext *ctx,
    double x,
    double y
)
{
    const double base = ctx->y_inf - y;
    const double x2 = x * x;
    const double y2 = y * y;
    const double y3 = y2 * y;

    const double radial_10 = pow_common(
        base,
        ctx->Gamma10
    );

    const double radial_2 = pow_common(
        base,
        ctx->Gamma2
    );

    return ctx->Gamma2 * (
        (x2 / y2) * radial_10 +
        (ctx->Gamma0 * x2 / y3) * radial_2 -
        radial_10
    );
}

/* Continuum luminosity absorbed by one shell.  When the ray crosses a second
   shell, transmission_2 accounts for attenuation before reaching this shell. */
static inline double get_Lshell(
    double x,
    double y,
    double tau_1,
    int red_side,
    const EmissionContext *ctx
)
{
    const double flux = interpolate_flux(ctx, x);
    if (flux == 0.0) return 0.0;
    const double shell = shell_factor(ctx, x, y);
    if (shell == 0.0) return 0.0;
    const double absorbed_1 = -expm1(-tau_1);
    if (absorbed_1 == 0.0) return 0.0;
    const double y_max = emission_y_max(ctx, x);

    if (y > y_max) {
        return -flux * shell * absorbed_1;
    }

    const double outer_boundary = ctx->y_inf - 1.0;
    const double x_outer = x / outer_boundary;

    const double h_s = sqrt(fmax(
        0.0,
        1.0 - x_outer * x_outer
    ));

    const double radial_max = pow_common(
        ctx->y_inf - y_max,
        ctx->Gamma2
    );

    const double x_over_y_max = x / y_max;

    const double h_max = sqrt(fmax(
        0.0,
        radial_max *
        (
            1.0 -
            x_over_y_max * x_over_y_max
        )
    ));

    const double radial = pow_common(
        ctx->y_inf - y,
        ctx->Gamma2
    );

    const double x_over_y = x / y;

    const double h = sqrt(fmax(
        0.0,
        radial *
        (
            1.0 -
            x_over_y * x_over_y
        )
    ));

    if (h > h_s && h < h_max) {
        const double y_2 = solve_emission_root(
            y_max,
            outer_boundary,
            x,
            ctx->Gamma2,
            ctx->y_inf,
            h
        );

        if (!isfinite(y_2)) return 0.0;

        const double f_g_2 = emission_geometry_scalar(
            ctx,
            y_2,
            x,
            red_side
        );

        double transmission_2 = 1.0;
        if (f_g_2 != 0.0) {
            const double base_2 = ctx->y_inf - y_2;
            const double x_over_y_2 = x / y_2;
            const double denominator_2 = y_2 +
                (ctx->Gamma0*base_2-y_2)*x_over_y_2*x_over_y_2;
            const double tau_2 = ctx->tau_0*
                pow_common(base_2,ctx->Gamma14)/denominator_2;
            transmission_2 -= f_g_2*(-expm1(-tau_2));
        }

        return (
            flux *
            shell *
            transmission_2 *
            absorbed_1
        );
    }

    return flux * shell * absorbed_1;
}

/* The four functions below return the fraction of an isovelocity ring that
   intersects the bicone for the same orientation cases used by absorption. */
static inline double emission_GeometryI_scalar(
    double y,
    double x,
    const EmissionContext *ctx
)
{
    const double projected = x / y;
    const double boundary_d = ctx->D;
    const double boundary_n = ctx->N;
    const double boundary_c = ctx->C * ctx->M - ctx->D;
    const double normalized_height = safe_sqrt(
        1.0 - projected * projected
    );

    double w, v, angle;
    double k_value, p_value, denominator;
    double f_g_lower, f_g_upper;

    if (projected > boundary_n) return 1.0;

    if (
        projected > boundary_d &&
        projected > boundary_c &&
        projected < boundary_n
    ) {
        w = ctx->M - ctx->D / ctx->C;
        v = (
            projected -
            w * ctx->C
        ) * ctx->Q;

        angle = acos(
            clamp_unit(v / normalized_height)
        );

        return 1.0 - angle * ctx->Gamma8;
    }

    if (
        projected > boundary_d &&
        projected < boundary_c &&
        projected < boundary_n
    ) {
        denominator = (
            ctx->M -
            ctx->D / ctx->E -
            projected / ctx->E
        );

        k_value = (
            ctx->D + projected
        ) / ctx->H;

        p_value = safe_sqrt(
            ctx->A * ctx->A -
            (ctx->A - k_value) *
            (ctx->A - k_value)
        );

        return (
            ctx->Gamma8 *
            atan(p_value / denominator)
        );
    }

    if (
        projected < boundary_d &&
        projected >= boundary_c &&
        projected < boundary_n
    ) {
        k_value = (
            ctx->D - projected
        ) / ctx->H;

        p_value = safe_sqrt(
            ctx->A * ctx->A -
            (ctx->A - k_value) *
            (ctx->A - k_value)
        );

        denominator = (
            ctx->M -
            ctx->D / ctx->E +
            projected / ctx->E
        );

        f_g_lower = (
            ctx->Gamma8 *
            atan(p_value / denominator)
        );

        w = ctx->M - ctx->D / ctx->C;

        v = (
            projected -
            w * ctx->C
        ) * ctx->Q;

        angle = acos(
            clamp_unit(v / normalized_height)
        );

        f_g_upper = 1.0 - angle * ctx->Gamma8;

        return f_g_upper + f_g_lower;
    }

    if (
        projected < boundary_d &&
        projected < boundary_c
    ) {
        k_value = (
            ctx->D - projected
        ) / ctx->H;

        p_value = safe_sqrt(
            ctx->A * ctx->A -
            (ctx->A - k_value) *
            (ctx->A - k_value)
        );

        denominator = (
            ctx->M -
            ctx->D / ctx->E +
            projected / ctx->E
        );

        f_g_lower = (
            ctx->Gamma8 *
            atan(p_value / denominator)
        );

        k_value = (
            ctx->D + projected
        ) / ctx->H;

        p_value = safe_sqrt(
            ctx->A * ctx->A -
            (ctx->A - k_value) *
            (ctx->A - k_value)
        );

        denominator = (
            ctx->M -
            ctx->D / ctx->E -
            projected / ctx->E
        );

        f_g_upper = (
            ctx->Gamma8 *
            atan(p_value / denominator)
        );

        return f_g_upper + f_g_lower;
    }

    return 0.0;
}

static inline double emission_GeometryII_scalar(
    double y,
    double x,
    const EmissionContext *ctx
)
{
    const double projected = x / y;
    const double boundary_r = ctx->R;
    const double boundary_upper = (
        ctx->Ig +
        ctx->O * ctx->E
    );
    const double normalized_height = safe_sqrt(
        1.0 - projected * projected
    );

    double v, angle;
    double b_value, d_value, h_value;

    if (projected >= boundary_r) return 1.0;

    if (
        projected > boundary_upper &&
        projected < boundary_r
    ) {
        v = (
            projected -
            boundary_upper
        ) * ctx->F;

        angle = acos(
            clamp_unit(v / normalized_height)
        );

        return 1.0 - angle * ctx->Gamma8;
    }

    if (
        projected > ctx->P &&
        projected < boundary_upper
    ) {
        b_value = (
            projected -
            ctx->Ig
        ) / ctx->H;

        d_value = safe_sqrt(
            ctx->A * ctx->A -
            (ctx->A - b_value) *
            (ctx->A - b_value)
        );

        h_value = (
            ctx->O -
            (projected - ctx->Ig) / ctx->E
        );

        return (
            ctx->Gamma8 *
            atan(d_value / h_value)
        );
    }

    return 0.0;
}

static inline double emission_GeometryIII_scalar(
    double y,
    double x,
    const EmissionContext *ctx
)
{
    const double projected = x / y;

    double denominator;
    double k_value;
    double p_value;
    double f_g_lower;
    double f_g_upper;

    if (
        projected >= ctx->D &&
        projected < ctx->N
    ) {
        denominator = (
            ctx->M -
            ctx->D / ctx->E -
            projected / ctx->E
        );

        k_value = (
            ctx->D +
            projected
        ) / ctx->H;

        p_value = safe_sqrt(
            ctx->A * ctx->A -
            (ctx->A - k_value) *
            (ctx->A - k_value)
        );

        return (
            ctx->Gamma8 *
            atan(p_value / denominator)
        );
    }

    if (projected < ctx->D) {
        denominator = (
            ctx->M -
            ctx->D / ctx->E -
            projected / ctx->E
        );

        k_value = (
            ctx->D +
            projected
        ) / ctx->H;

        p_value = safe_sqrt(
            ctx->A * ctx->A -
            (ctx->A - k_value) *
            (ctx->A - k_value)
        );

        f_g_upper = (
            ctx->Gamma8 *
            atan(p_value / denominator)
        );

        k_value = (
            ctx->D -
            projected
        ) / ctx->H;

        p_value = safe_sqrt(
            ctx->A * ctx->A -
            (ctx->A - k_value) *
            (ctx->A - k_value)
        );

        denominator = (
            ctx->M -
            ctx->D / ctx->E +
            projected / ctx->E
        );

        f_g_lower = (
            ctx->Gamma8 *
            atan(p_value / denominator)
        );

        return f_g_lower + f_g_upper;
    }

    return 0.0;
}

static inline double emission_GeometryIV_scalar(
    double y,
    double x,
    const EmissionContext *ctx
)
{
    const double projected = x / y;

    if (
        projected > ctx->P &&
        projected < ctx->R
    ) {
        const double p_value = (
            projected -
            ctx->P
        ) / ctx->H;

        const double u_value = safe_sqrt(
            ctx->A * ctx->A -
            (ctx->A - p_value) *
            (ctx->A - p_value)
        );

        const double denominator = (
            ctx->O -
            p_value * ctx->G
        );

        return (
            ctx->Gamma8 *
            atan(u_value / denominator)
        );
    }

    return 0.0;
}

/* Dispatch the projection geometry, then remove far-side rays occulted by
   the source and rays falling outside the observing aperture. */
static inline double emission_geometry_scalar(
    const EmissionContext *ctx,
    double y,
    double x,
    int red_side
)
{
    double geometry;
    switch (ctx->geometry) {
        case 0:
            geometry = 1.0;
            break;

        case 1:
            geometry = emission_GeometryI_scalar(
                y,
                x,
                ctx
            );
            break;

        case 2:
            geometry = emission_GeometryII_scalar(
                y,
                x,
                ctx
            );
            break;

        case 3:
            geometry = emission_GeometryIII_scalar(
                y,
                x,
                ctx
            );
            break;

        case 4:
            geometry = emission_GeometryIV_scalar(
                y,
                x,
                ctx
            );
            break;

        default:
            return 0.0;
    }
    const double q = x / y;
    const double h = sqrt(fmax(0.0, pow_common(ctx->y_inf - y, ctx->Gamma2) * (1.0 - q * q)));
    if (ctx->occultation && red_side && h < 1.0) return 0.0;
    if (ctx->aperture && h > ctx->r_ap) return 0.0;
    return geometry;
}

/* Inner quadrature: integrate the continuum absorbed by shell y over all
   incident projected velocities that resonate within that shell. */
static double integrate_shell_luminosity(
    double y,
    double theta_c,
    double tau_1,
    int red_side,
    const EmissionContext *ctx
)
{
    const double lower = y * cos(theta_c);
    const double upper = y;

    if (!(upper > lower)) return 0.0;

    double result = 0.0;
    const double midpoint = 0.5 * (lower + upper);
    const double half_width = 0.5 * (upper - lower);
    for (int i = 0; i < ctx->inner_points; ++i) {
        const double x_shell = midpoint + half_width * ctx->inner_node[i];
        const double weight = half_width * ctx->inner_weight[i];
        result += weight * get_Lshell(
            x_shell, y, tau_1, red_side, ctx
        );
    }
    return result;
}

/* Contribution of shell y to the observed emission at x, including escape
   probability, bicone coverage, dust, covering fraction, and 1/(2y)
   redistribution over projected velocity. */
static inline double emission_integrand(
    double y,
    double x,
    const EmissionContext *ctx
)
{
    if (ctx->f_c == 0.0) return 0.0;
    const int red_side = x < 0.0;
    /* Emission_Integral already restricts spherical rays to the exact
       occultation/aperture support, so no per-node geometry calculation is
       needed for a full sphere. */
    const double geometry = ctx->geometry == 0 ? 1.0 :
        emission_geometry_scalar(ctx, y, fabs(x), red_side);
    if (geometry == 0.0) return 0.0;

    const double base = ctx->y_inf - y;
    const double theta_argument = clamp_unit(
        pow_common(base, ctx->Gamma18)
    );

    const double theta_c = asin(theta_argument);
    const double cos_theta_c = cos(theta_c);

    double theta_average;

    if (fabs(theta_c) < 1.0e-12) {
        theta_average = 1.0;
    } else {
        theta_average = (
            theta_c +
            cos_theta_c * theta_argument
        ) / (2.0 * theta_c);
    }

    const double tau_denominator = (
        y +
        (
            ctx->Gamma0 * base - y
        ) * theta_average
    );

    const double tau_1 = (
        ctx->tau_0 *
        pow_common(base, ctx->Gamma14) /
        tau_denominator
    );

    const double shell_luminosity = integrate_shell_luminosity(
        y,
        theta_c,
        tau_1,
        red_side,
        ctx
    );
    if (shell_luminosity == 0.0) return 0.0;

    double scattering_fraction = 1.0;

    if (ctx->resonance || ctx->fluorescence) {
        const double beta = getBeta(
            y,
            ctx->tau_0,
            ctx->y_inf,
            ctx->Gamma4,
            ctx->Gamma9,
            ctx->Gamma17,
            ctx->change_approx
        );

        const double denominator = (
            1.0 -
            ctx->p_r * (1.0 - beta)
        );

        if (denominator == 0.0) return 0.0;

        if (ctx->resonance) {
            scattering_fraction = (
                beta *
                ctx->p_r /
                denominator
            );
        } else {
            scattering_fraction = (
                ctx->p_f /
                denominator
            );
        }
    }

    if (scattering_fraction == 0.0) return 0.0;
    const double tau_d = dust_scalar(
        y, x, ctx->y_inf, ctx->k_dust, ctx->delta,
        ctx->Gamma3, ctx->Gamma5, ctx->Gamma12, ctx->Gamma13,
        ctx->Gamma14, ctx->Gamma15, ctx->Gamma16
    );

    return (
        exp(-tau_d) *
        scattering_fraction *
        geometry *
        ctx->f_c *
        shell_luminosity /
        (2.0 * y)
    );
}

/* Outer quadrature over emitting shells at one observed velocity.  Bounds are
   clipped analytically for occultation, aperture, and empty geometry support
   before any expensive integrand evaluations are attempted. */
static double Emission_Integral(
    double x,
    const EmissionContext *ctx
)
{
    const double abs_x = fabs(x);
    const double outer_boundary = ctx->y_inf - 1.0;

    double lower_bound;
    double upper_bound;

    if (x < 0.0 && ctx->occultation) {
        const double y_max = emission_y_max(
            ctx,
            abs_x
        );

        const double radial_max = pow_common(
            ctx->y_inf - y_max,
            ctx->Gamma2
        );

        const double x_over_y_max = abs_x / y_max;

        const double h_max = safe_sqrt(
            radial_max *
            (
                1.0 -
                x_over_y_max * x_over_y_max
            )
        );

        if (h_max < 1.0) return 0.0;

        lower_bound = solve_emission_root(
            abs_x,
            y_max,
            abs_x,
            ctx->Gamma2,
            ctx->y_inf,
            1.0
        );

        upper_bound = solve_emission_root(
            y_max,
            ctx->y_inf - 1.0e-3,
            abs_x,
            ctx->Gamma2,
            ctx->y_inf,
            1.0
        );

        upper_bound = fmin(
            upper_bound,
            outer_boundary
        );

        if (
            !isfinite(lower_bound) ||
            !isfinite(upper_bound)
        ) {
            return 0.0;
        }
    } else {
        lower_bound = abs_x;
        upper_bound = outer_boundary;
    }

    /* Clip to the exact nonzero support of the piecewise geometry.  The
       common radial factor cancels from each boundary inequality, leaving
       simple limits in q=|x|/y.  This both resolves narrow line-center
       contributions and avoids quadrature work where f_g is identically 0. */
    if ((ctx->geometry == 2 || ctx->geometry == 4) && ctx->P > 0.0) {
        upper_bound = fmin(upper_bound, abs_x / ctx->P);
    } else if (ctx->geometry == 3 && ctx->N > 0.0) {
        lower_bound = fmax(lower_bound, abs_x / ctx->N);
    }

    if (!(upper_bound > lower_bound)) return 0.0;

    double intervals[4] = {lower_bound, upper_bound, 0.0, 0.0};
    int n_intervals = 1;

    if (ctx->aperture && ctx->r_ap < 1.0) {
        /* On the red side occultation admits h>=1 while a sub-source aperture
           admits h<=r_ap<1, so their intersection is empty. */
        if (x < 0.0 && ctx->occultation) return 0.0;

        const double y_max = emission_y_max(ctx, abs_x);
        const double radial_max = pow_common(ctx->y_inf-y_max, ctx->Gamma2);
        const double q_max = abs_x/y_max;
        const double h_max = safe_sqrt(radial_max*(1.0-q_max*q_max));

        if (h_max > ctx->r_ap) {
            const double left = solve_emission_root(
                abs_x, y_max, abs_x, ctx->Gamma2, ctx->y_inf, ctx->r_ap
            );
            const double right = solve_emission_root(
                y_max, outer_boundary, abs_x, ctx->Gamma2,
                ctx->y_inf, ctx->r_ap
            );
            n_intervals = 0;
            if (isfinite(left)) {
                const double end = fmin(upper_bound,left);
                if (end > lower_bound) {
                    intervals[2*n_intervals] = lower_bound;
                    intervals[2*n_intervals+1] = end;
                    ++n_intervals;
                }
            }
            if (isfinite(right)) {
                const double start = fmax(lower_bound,right);
                if (upper_bound > start) {
                    intervals[2*n_intervals] = start;
                    intervals[2*n_intervals+1] = upper_bound;
                    ++n_intervals;
                }
            }
        }
    }

    double result = 0.0;
    for (int interval = 0; interval < n_intervals; ++interval) {
        const double start = intervals[2*interval];
        const double end = intervals[2*interval+1];
        const double midpoint = 0.5*(start+end);
        const double half_width = 0.5*(end-start);
        for (int i = 0; i < ctx->outer_points; ++i) {
            const double y = midpoint+half_width*ctx->outer_node[i];
            const double weight = half_width*ctx->outer_weight[i];
            result += weight*emission_integrand(y,x,ctx);
        }
    }
    return result;
}

/* Locate the shell coordinate at which the Bernoulli and optically thick
   escape-probability approximations should switch.  A zero return requests a
   single approximation over the entire wind. */
static double find_change_approx(
    const EmissionContext *ctx
)
{
    enum { N_CHANGE = 100 };

    const double dy = (
        ctx->y_inf - 1.0
    ) / (N_CHANGE - 1);

    double previous = 0.0;
    double current = 0.0;
    double next = 0.0;

    for (int i = 0; i < N_CHANGE; ++i) {
        const double y = 1.0 + i * dy;

        const double difference = (
            getBernoulli3(
                y,
                ctx->tau_0,
                ctx->y_inf,
                ctx->Gamma4,
                ctx->Gamma9
            ) -
            getAsymptotic(
                y,
                ctx->tau_0,
                ctx->y_inf,
                ctx->Gamma4,
                ctx->Gamma9
            )
        );

        if (i == 0) {
            previous = difference;
            continue;
        }

        if (i == 1) {
            current = difference;
            continue;
        }

        next = difference;

        if (current > previous && current > next) {
            return 1.0 + (i - 1) * dy;
        }

        previous = current;
        current = next;
    }

    return 0.0;
}

/* Nearest-bin lookup used only while aligning the incident continuum with the
   absorbing transition. */
static size_t nearest_velocity_index(
    const double *v_obs,
    size_t n,
    double target
)
{
    size_t best = 0;
    double best_distance = fabs(v_obs[0] - target);
    for (size_t i = 1; i < n; ++i) {
        const double distance = fabs(v_obs[i] - target);
        if (distance < best_distance) { best = i; best_distance = distance; }
    }
    return best;
}

/* Shift a sampled spectrum by an integer number of bins and zero-fill exposed
   edges; wrapping would incorrectly mix opposite ends of the velocity grid. */
static void shift_flux_without_wrap(
    const double *source,
    double *destination,
    size_t n,
    long shift
)
{
    for (size_t i = 0; i < n; ++i) {
        const long source_index = (long)i - shift;
        destination[i] = (source_index >= 0 && source_index < (long)n)
                       ? source[(size_t)source_index] : 0.0;
    }
}

/*
 * Compute one resonant or fluorescent emission channel.
 *
 * normalized_flux is the continuum incident on the parent transition after
 * all preceding absorption stages.  wavelength and lambda_ref are in
 * Angstrom; velocities are in km s^-1.  emission receives the additive,
 * continuum-normalized line contribution on the original v_obs grid.
 */
void computeEM_Inflow(
    double wavelength,
    double oscillator_strength,
    double lambda_ref,
    const double *v_obs,
    const double *normalized_flux,
    size_t n_velocity,
    double alpha,
    double psi,
    double gamma,
    double tau,
    double v_0,
    double v_w,
    double v_ap,
    double f_c,
    double k_dust,
    double delta,
    int resonance,
    int fluorescence,
    int occultation,
    int aperture,
    double p_r,
    double p_f,
    double *emission
)
{
    EmissionContext ctx = {0};

    /* Cache all velocity-independent physics and geometry. */
    ctx.tau_0 = wavelength * oscillator_strength * tau;
    ctx.y_inf = v_w / v_0;
    double y_ap = v_ap / v_0;
    if (y_ap > ctx.y_inf) y_ap = ctx.y_inf;
    ctx.f_c = f_c;
    ctx.k_dust = k_dust;
    ctx.delta = delta;
    ctx.p_r = p_r;
    ctx.p_f = p_f;
    ctx.resonance = resonance;
    ctx.fluorescence = fluorescence;
    ctx.occultation = occultation;
    ctx.aperture = aperture;

    ctx.A = sin(alpha);
    ctx.C = tan(alpha - fabs(psi - alpha));
    ctx.D = sin(psi + alpha - M_PI / 2.0);
    ctx.E = tan(psi);
    ctx.F = tan(M_PI / 2.0 - psi);
    ctx.G = cos(psi);
    ctx.H = sin(psi);
    ctx.Ig = cos(psi + alpha);
    ctx.M = cos(psi + alpha - M_PI / 2.0);
    ctx.N = cos(fabs(psi - alpha));
    ctx.O = sin(alpha + psi);
    ctx.P = cos(alpha + psi);
    ctx.Q = tan(M_PI / 2.0 - alpha + fabs(psi - alpha));
    ctx.R = cos(psi - alpha);

    ctx.Gamma0 = gamma;
    ctx.Gamma2 = 2.0 / gamma;
    ctx.Gamma3 = 1.0 / gamma;
    ctx.Gamma4 = (1.0 - delta - gamma) / gamma;
    ctx.Gamma5 = (delta != 1.0) ? 1.0 / (1.0 - delta) : 1.0;
    ctx.Gamma8 = 1.0 / M_PI;
    ctx.Gamma9 = gamma - 1.0;
    ctx.Gamma10 = (2.0 - gamma) / gamma;
    ctx.Gamma12 = pow_common(ctx.y_inf, 1.0 / gamma);
    ctx.Gamma13 = pow_common(ctx.y_inf, (1.0 - delta) / gamma);
    ctx.Gamma14 = (1.0 - delta) / gamma;
    ctx.Gamma15 = delta / 2.0;
    ctx.Gamma16 = 2.0 + gamma;
    ctx.Gamma17 = ctx.tau_0 * pow_common(ctx.y_inf, ctx.Gamma4);
    ctx.Gamma18 = -1.0 / gamma;
    ctx.Gamma20 = -gamma * ctx.y_inf;
    ctx.r_ap = pow_common(ctx.y_inf - y_ap, ctx.Gamma3);

    /* A sphere has unit geometry and uses a lower-cost quadrature rule. */
    if (fabs(alpha - M_PI / 2.0) < 1.0e-14) {
        ctx.geometry = 0;
    } else if (alpha + psi > M_PI / 2.0 && psi - alpha <= 0.0) {
        ctx.geometry = 1;
    } else if (
        alpha + psi <= M_PI / 2.0 &&
        psi - alpha <= 0.0
    ) {
        ctx.geometry = 2;
    } else if (
        alpha + psi > M_PI / 2.0 &&
        psi - alpha > 0.0
    ) {
        ctx.geometry = 3;
    } else {
        ctx.geometry = 4;
    }

    ctx.outer_points = ctx.geometry == 0
                     ? SPHERICAL_OUTER_POINTS : EMISSION_OUTER_POINTS;
    ctx.inner_points = ctx.geometry == 0
                     ? SPHERICAL_INNER_POINTS : EMISSION_INNER_POINTS;

    gsl_integration_glfixed_table *outer_quad =
        gsl_integration_glfixed_table_alloc((size_t)ctx.outer_points);
    gsl_integration_glfixed_table *inner_quad =
        gsl_integration_glfixed_table_alloc((size_t)ctx.inner_points);

    if (
        outer_quad == NULL ||
        inner_quad == NULL
    ) {
        for (size_t i = 0; i < n_velocity; ++i) {
            emission[i] = NAN;
        }

        gsl_integration_glfixed_table_free(outer_quad);
        gsl_integration_glfixed_table_free(inner_quad);
        return;
    }

    /* Copy the nodes into the context once; GSL tables are not touched by the
       parallel velocity loop. */
    for (int i = 0; i < ctx.outer_points; ++i) {
        gsl_integration_glfixed_point(
            -1.0, 1.0, (size_t)i,
            &ctx.outer_node[i], &ctx.outer_weight[i], outer_quad
        );
    }
    for (int i = 0; i < ctx.inner_points; ++i) {
        gsl_integration_glfixed_point(
            -1.0, 1.0, (size_t)i,
            &ctx.inner_node[i], &ctx.inner_weight[i], inner_quad
        );
    }
    gsl_integration_glfixed_table_free(outer_quad);
    gsl_integration_glfixed_table_free(inner_quad);

    double *shifted_flux = calloc(n_velocity, sizeof(*shifted_flux));
    double *flux_x = malloc(n_velocity * sizeof(*flux_x));
    double *flux_sorted = malloc(n_velocity * sizeof(*flux_sorted));

    if (shifted_flux == NULL || flux_x == NULL || flux_sorted == NULL ||
        normalized_flux == NULL || v_obs == NULL || n_velocity == 0 || v_0 == 0.0) {
        for (size_t i = 0; i < n_velocity; ++i) emission[i] = NAN;
        free(shifted_flux); free(flux_x); free(flux_sorted);
        return;
    }

    /* Shift the incident spectrum from lambda_ref to the parent transition. */
    const double speed_of_light = 2.99792458e5;
    const double velocity_shift = -speed_of_light * (wavelength - lambda_ref) / lambda_ref;
    const long shift = (long)nearest_velocity_index(v_obs, n_velocity, velocity_shift)
                     - (long)nearest_velocity_index(v_obs, n_velocity, 0.0);
    shift_flux_without_wrap(normalized_flux, shifted_flux, n_velocity, shift);

    /* Interpolation requires increasing dimensionless coordinates.  Because
       internal inflow v_0 is negative, an increasing v_obs array is commonly
       reversed here. */
    const int reverse = (v_obs[0] / v_0) > (v_obs[n_velocity - 1] / v_0);
    for (size_t i = 0; i < n_velocity; ++i) {
        const size_t source = reverse ? n_velocity - 1 - i : i;
        flux_x[i] = v_obs[source] / v_0;
        flux_sorted[i] = shifted_flux[source];
    }

    ctx.flux_x = flux_x;
    ctx.flux = flux_sorted;
    ctx.n_flux = n_velocity;
    /* Detect constant and uniformly spaced inputs to avoid binary searches in
       the nested quadrature loops. */
    ctx.flux_constant = 1;
    ctx.flux_constant_value = flux_sorted[0];
    for (size_t i = 1; i < n_velocity; ++i) {
        if (flux_sorted[i] != ctx.flux_constant_value) {
            ctx.flux_constant = 0;
            break;
        }
    }
    ctx.flux_uniform = 0;
    if (n_velocity > 1) {
        const double step = flux_x[1] - flux_x[0];
        if (step > 0.0) {
            const double tolerance = 64.0 * 2.2204460492503131e-16
                                   * fmax(1.0, fabs(flux_x[n_velocity - 1]));
            ctx.flux_uniform = 1;
            for (size_t i = 2; i < n_velocity; ++i) {
                if (fabs(flux_x[i] - (flux_x[0] + step * (double)i)) > tolerance) {
                    ctx.flux_uniform = 0;
                    break;
                }
            }
            if (ctx.flux_uniform) {
                ctx.flux_x0 = flux_x[0];
                ctx.flux_inv_step = 1.0 / step;
            }
        }
    }

    ctx.change_approx = find_change_approx(&ctx);

    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (size_t i = 0; i < n_velocity; ++i) {
        double x = -v_obs[i] / v_0;

        if (x == 0.0) {
            const double epsilon = 1.0e-3;
            emission[i] = 0.5 * (
                Emission_Integral(epsilon, &ctx) +
                Emission_Integral(-epsilon, &ctx)
            );
            continue;
        }
        if (x > 0.0 && x < 1.0e-3) x = 1.0e-3;
        if (x <= 0.0 && x > -1.0e-3) x = -1.0e-3;

        if (
            fabs(x) > ctx.y_inf - 1.0 ||
            alpha == 0.0 ||
            ctx.tau_0 == 0.0
        ) {
            emission[i] = 0.0;
        } else {
            emission[i] = Emission_Integral(
                x,
                &ctx
            );
        }
    }

    /* At exactly zero velocity the analytic support can be narrower than
       the smallest fixed quadrature node.  Use the two-sided grid limit
       instead of assigning an arbitrary red- or blue-side epsilon. */
    for (size_t i = 1; i + 1 < n_velocity; ++i) {
        if (v_obs[i] == 0.0) {
            const double width = v_obs[i + 1] - v_obs[i - 1];
            emission[i] = width != 0.0
                        ? (emission[i - 1] * (v_obs[i + 1] - v_obs[i]) +
                           emission[i + 1] * (v_obs[i] - v_obs[i - 1])) / width
                        : 0.5 * (emission[i - 1] + emission[i + 1]);
        }
    }

    free(shifted_flux);
    free(flux_x);
    free(flux_sorted);
}
