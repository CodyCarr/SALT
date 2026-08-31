# SPDX-License-Identifier: BSD-3-Clause
"""Minimal, self-contained C SALT inflow example."""

import time

import matplotlib.pyplot as plt
import numpy as np

from run_salt import salt


# Center the velocity grid on Fe II 2343.49.  The red side extends far enough
# to include its Fe II* fluorescent channels at 2364.83 and 2380.76 Angstrom.
v_obs = np.linspace(-1000.0, 5500.0, 1000)
lam_ref = 2343.49
background = np.ones_like(v_obs)

# Inflow speeds are supplied as positive magnitudes.  The C dispatcher applies
# the internal inflow sign convention.
flow_parameters = {
    "alpha": np.pi / 4.0,
    "psi": 0.0,
    "gamma": 1.0,
    "tau": 0.1,
    "v_0": 25.0,
    "v_w": 500.0,
    "f_c": 1.0,
    "k": 0.0,
    "delta": 1.5,
}

absorption_parameters = {
    "abs_waves": [2343.49],
    "abs_osc_strs": [0.114],
}

# Branching probabilities are A_ul / sum(A_ul), using A_ul = 1.73e8,
# 5.90e7, and 3.10e7 s^-1 for the three decays from the shared upper level.
emission_parameters = {
    "em_waves": [2343.49, 2343.49, 2343.49],
    "emitted_waves": [2343.49, 2364.83, 2380.76],
    "em_osc_strs": [0.114, 0.114, 0.114],
    "res": [True, False, False],
    "fluor": [False, True, True],
    "p_r": [0.6578, 0.6578, 0.6578],
    "p_f": [0.3422, 0.2243, 0.1179],
    "line_num": [3],
}

observing_parameters = {"APERTURE": True, "v_ap": 400.0, "v_obs": v_obs}
miscellaneous_parameters = {"OCCULTATION": True, "lam_ref": lam_ref}

profile_parameters = {
    "absorption_parameters": absorption_parameters,
    "emission_parameters": emission_parameters,
    "observing_parameters": observing_parameters,
    "miscellaneous_parameters": miscellaneous_parameters,
}

arguments = {
    "v_obs": v_obs,
    "lam_ref": lam_ref,
    "background": background,
    "flow_parameters": flow_parameters,
    "profile_parameters": profile_parameters,
    "profile_type": "pcygni",
    "model_type": "inflow",
}

# Warm up shared-library and OpenMP initialization before benchmarking.
salt(**arguments)
times = []
for _ in range(10):
    start = time.perf_counter()
    spectrum = salt(**arguments)
    times.append(time.perf_counter() - start)

print(f"Median runtime: {np.median(times):.6f} s")
print(f"Flux range: {spectrum.min():.6f} to {spectrum.max():.6f}")

# plot spectrum

c_kms = 299792.458
ymax = 1.1*np.max(spectrum)
ymin = -0.2
line_waves = np.array([2343.49, 2364.83, 2380.76])
zeros = c_kms * (line_waves / lam_ref - 1.0)

fig, ax = plt.subplots(figsize=(7, 5))

ax.vlines(zeros[[0]], ymin, ymax, colors="b", linestyles="-", alpha=0.6)
ax.vlines(zeros[[1, 2]], ymin, ymax, colors="b", linestyles="--", alpha=0.6)
ax.plot(v_obs, spectrum, color="tab:red", linewidth=2)

ax.axhline(1.0, color="0.6", linestyle="--")
ax.set(xlabel=r"Observed velocity [km s$^{-1}$]", ylabel=r"$F/F_0$")
ax.set_xlim(-1000,5500)
ax.set_ylim(ymin,ymax)
ax.grid()
fig.tight_layout()
plt.show()
