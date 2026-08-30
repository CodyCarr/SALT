/* SPDX-License-Identifier: BSD-3-Clause */

#pragma once

/*
 * Compile-time numerical resolutions used by the three SALT kernels.
 *
 * Change grid sizes here rather than in the individual .c files.  The
 * absorption, emission, and line-profile grids serve different purposes, so
 * they intentionally have separate names and do not need equal resolutions.
 */
enum {
    SALT_ABS_NS = 96,
    SALT_ABS_NPHI = 96,
    SALT_ABS_NY = 65,
    SALT_ABS_NY_SOBOLEV = 65,

    SALT_EM_NB = 96,
    SALT_EM_NY = 256,
    SALT_EM_NS = 65,

    SALT_LINE_NS = 10,
    SALT_LINE_NYO = 1000,
    SALT_LINE_FIRST_SHELL_OFFSET = 65536,

    SALT_EM_FIRST_FLUX_OFFSET = 4097
};

/* Every sampled integration needs at least two endpoints. */
_Static_assert(SALT_ABS_NS >= 3, "SALT_ABS_NS must be at least 3");
_Static_assert(SALT_ABS_NPHI >= 8, "SALT_ABS_NPHI must be at least 8");
_Static_assert(SALT_ABS_NY >= 3, "SALT_ABS_NY must be at least 3");
_Static_assert(SALT_ABS_NY_SOBOLEV >= 3,
               "SALT_ABS_NY_SOBOLEV must be at least 3");
_Static_assert(SALT_EM_NB >= 3, "SALT_EM_NB must be at least 3");
_Static_assert(SALT_EM_NY >= 3, "SALT_EM_NY must be at least 3");
_Static_assert(SALT_EM_NS >= 2, "SALT_EM_NS must be at least 2");
_Static_assert(SALT_LINE_NS >= 2, "SALT_LINE_NS must be at least 2");
_Static_assert(SALT_LINE_NYO >= 2, "SALT_LINE_NYO must be at least 2");
_Static_assert(SALT_LINE_FIRST_SHELL_OFFSET > 0,
               "SALT_LINE_FIRST_SHELL_OFFSET must be positive");
_Static_assert(SALT_EM_FIRST_FLUX_OFFSET > 0,
               "SALT_EM_FIRST_FLUX_OFFSET must be positive");
