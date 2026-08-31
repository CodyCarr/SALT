/* SPDX-License-Identifier: BSD-3-Clause */

/*
 * Inflow-only SALT line-profile assembly.
 *
 * The inflow formalism follows Carr & Scarlata (2022), ApJ, 939, 47,
 * doi:10.3847/1538-4357/ac93fa.
 *
 * This file is the orchestration layer: it applies absorption lines in their
 * physical order, retains the continuum incident on each emission stage,
 * computes every resonant/fluorescent channel, shifts each transition from its
 * own rest wavelength onto the reference-velocity grid, and finally combines
 * the requested profile components.
 */

#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include "salt.h"

static const double C_KM_S = 2.99792458e5;

void computeABS_Inflow(
    double wavelength, double oscillator_strength,
    const double *v_obs, size_t n_velocity,
    double alpha, double psi, double gamma, double tau,
    double v_0, double v_w, double v_ap, double f_c, double delta, int aperture,
    double *absorption
);

void computeEM_Inflow(
    double wavelength, double oscillator_strength, double lambda_ref,
    const double *v_obs, const double *normalized_flux, size_t n_velocity,
    double alpha, double psi, double gamma, double tau,
    double v_0, double v_w, double v_ap, double f_c, double k_dust, double delta,
    int resonance, int fluorescence, int occultation, int aperture,
    double p_r, double p_f, double *emission
);

/* Binary search for the grid point nearest value.  v_obs must be increasing. */
static size_t nearest_index(const double *x, size_t n, double value)
{
    if (value <= x[0]) return 0;
    if (value >= x[n - 1]) return n - 1;

    size_t lower = 0;
    size_t upper = n - 1;
    while (upper - lower > 1) {
        const size_t middle = lower + (upper - lower) / 2;
        if (x[middle] < value) lower = middle;
        else upper = middle;
    }

    return fabs(value - x[lower]) <= fabs(x[upper] - value) ? lower : upper;
}

/* Match the legacy nearest-bin shift while zero-filling exposed edges. */
static void shift_bins_zero(
    double *destination, const double *source,
    const double *v_obs, size_t n, double velocity_shift
)
{
    const ptrdiff_t shift = (ptrdiff_t)nearest_index(v_obs, n, velocity_shift)
                            - (ptrdiff_t)nearest_index(v_obs, n, 0.0);

    for (size_t i = 0; i < n; ++i) {
        const ptrdiff_t source_index = (ptrdiff_t)i - shift;
        destination[i] = source_index >= 0 && (size_t)source_index < n
                       ? source[(size_t)source_index] : 0.0;
    }
}

/* NaN is the C API's explicit failure signal for an invalid request. */
static void fill_nan(double *values, size_t n)
{
    for (size_t i = 0; i < n; ++i) values[i] = NAN;
}

/* Map a flattened emission channel to the absorption stage that provides its
   incident spectrum. */
static int emission_stage(const int *line_num, size_t n_line_num, size_t line)
{
    size_t first = 0;
    for (size_t stage = 0; stage < n_line_num; ++stage) {
        if (line_num[stage] < 0) return -1;
        const size_t count = (size_t)line_num[stage];
        if (line < first + count) return (int)stage;
        first += count;
    }
    return -1;
}

/* See include/salt.h for the public contract and units. */
SALT_API void Line_Profile_Inflow(
    const double *v_obs, size_t n_velocity,
    double lambda_ref, const double *background,
    const double *abs_waves, const double *abs_osc, size_t n_abs,
    const double *em_waves, const double *emitted_waves,
    const double *em_osc, size_t n_em,
    const int *resonance, const int *fluorescence,
    const double *p_r, const double *p_f,
    const int *line_num, size_t n_line_num,
    double alpha, double psi, double gamma, double tau,
    double v_0, double v_w, double v_ap, double f_c, double k_dust, double delta,
    int aperture, int occultation, int profile_type,
    double *output
)
{
    if (output == NULL || v_obs == NULL || background == NULL ||
        n_velocity == 0 || profile_type < 0 || profile_type > 2 ||
        v_0 == 0.0 || lambda_ref == 0.0 ||
        (aperture && fabs(v_ap) >= fabs(v_w)) ||
        (n_abs > 0 && (abs_waves == NULL || abs_osc == NULL))) {
        if (output != NULL) fill_nan(output, n_velocity);
        return;
    }

    if (profile_type != 0 && n_em > 0 &&
        (em_waves == NULL || emitted_waves == NULL || em_osc == NULL ||
         resonance == NULL || fluorescence == NULL || p_r == NULL ||
         p_f == NULL || line_num == NULL || n_line_num == 0)) {
        fill_nan(output, n_velocity);
        return;
    }

    /* Store the input continuum and the spectrum after each absorption line.
       Emission from a transition must see only the absorption stages that
       precede its parent line.  kernel and shifted reuse the same allocation. */
    const size_t stage_values = (n_abs + 1U) * n_velocity;
    const size_t workspace_values = stage_values + 2U * n_velocity;
    double *workspace = malloc(workspace_values * sizeof(*workspace));
    if (workspace == NULL) {
        fill_nan(output, n_velocity);
        return;
    }

    double *stages = workspace;
    double *kernel = workspace + stage_values;
    double *shifted = kernel + n_velocity;
    memcpy(stages, background, n_velocity * sizeof(*stages));

    for (size_t line = 0; line < n_abs; ++line) {
        /* The public Python interface uses positive speed magnitudes.  The
           inflow solvers use negative radial velocities internally. */
        computeABS_Inflow(
            abs_waves[line], abs_osc[line], v_obs, n_velocity,
            alpha, psi, gamma, tau, -v_0, -v_w, -v_ap,
            f_c, delta, aperture, kernel
        );

        /* Non-relativistic wavelength-to-velocity offset relative to the
           common reference transition. */
        const double velocity_shift = C_KM_S
                                    * (abs_waves[line] - lambda_ref)
                                    / lambda_ref;
        shift_bins_zero(shifted, kernel, v_obs, n_velocity, velocity_shift);

        const double *previous = stages + line * n_velocity;
        double *current = stages + (line + 1U) * n_velocity;
        for (size_t i = 0; i < n_velocity; ++i) {
            current[i] = previous[i] * (1.0 + shifted[i]);
        }
    }

    const double *final_absorption = stages + n_abs * n_velocity;
    if (profile_type == 0) {
        memcpy(output, final_absorption, n_velocity * sizeof(*output));
        free(workspace);
        return;
    }

    /* Emission channels add linearly after being shifted to their final
       resonant or fluorescent wavelength. */
    memset(output, 0, n_velocity * sizeof(*output));
    for (size_t line = 0; line < n_em; ++line) {
        const int stage = emission_stage(line_num, n_line_num, line);
        if (stage < 0 || (size_t)stage > n_abs) {
            fill_nan(output, n_velocity);
            free(workspace);
            return;
        }

        computeEM_Inflow(
            em_waves[line], em_osc[line], lambda_ref,
            v_obs, stages + (size_t)stage * n_velocity, n_velocity,
            alpha, psi, gamma, tau, -v_0, -v_w, -v_ap,
            f_c, k_dust, delta,
            resonance[line], fluorescence[line], occultation, aperture,
            p_r[line], p_f[line], kernel
        );

        const double velocity_shift = C_KM_S
                                    * (emitted_waves[line] - lambda_ref)
                                    / lambda_ref;
        shift_bins_zero(shifted, kernel, v_obs, n_velocity, velocity_shift);
        for (size_t i = 0; i < n_velocity; ++i) output[i] += shifted[i];
    }

    if (profile_type == 2) {
        for (size_t i = 0; i < n_velocity; ++i) output[i] += final_absorption[i];
    }

    free(workspace);
}
