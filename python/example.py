# SPDX-License-Identifier: BSD-3-Clause
"""Minimal, self-contained C SALT example."""

import time

import matplotlib.pyplot as plt
import numpy as np

from run_salt import salt


v_obs = np.linspace(-1000.0, 1000.0, 1000)
lam_ref = 1190.42
background = np.ones_like(v_obs)

flow_parameters = {
    "alpha": np.pi / 2.0,
    "psi": 0.0,
    "gamma": 1.0,
    "tau": 0.1,
    "v_0": 25.0,
    "v_w": 500.0,
    "v_b": 10.0,
    "f_c": 1.0,
    "k": 0.0,
    "delta": 1.5,
}

absorption_parameters = {
    "abs_waves": [1190.42],
    "abs_osc_strs": [0.277],
    "abs_ein": [6.53e8],
}

emission_parameters = {
    "em_waves": [1190.42],
    "emitted_waves": [1190.42],
    "em_osc_strs": [0.277],
    "em_ein": [6.53e8],
    "res": [True],
    "fluor": [False],
    "p_r": [1.0],
    "p_f": [0.0],
    "line_num": [1],
}

# Each emission channel needs a blending entry. The flag is False here, so
# the zero-valued placeholders are packed but never used by the C kernel.
blending_parameters = {
    "blended_waves": [[0.0]],
    "blended_osc_strs": [[0.0]],
    "blended_abs_ein": [[0.0]],
    "blended_fluor": [[False]],
    "blended_p_r": [[0.0]],
    "blended_p_f": [[[0.0]]],
    "blended_flour_waves": [[[0.0]]],
    "blending": [False],
}

observing_parameters = {"APERTURE": True, "v_ap": 500.0, "v_obs": v_obs}
miscellaneous_parameters = {
    "OCCULTATION": True,
    "lam_ref": lam_ref,
    "Sobolev": True,
    "SW": 100.0,
    "profile_method": "colt",
}

profile_parameters = {
    "absorption_parameters": absorption_parameters,
    "emission_parameters": emission_parameters,
    "blending_parameters": blending_parameters,
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

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(v_obs, spectrum, color="black", linewidth=2)
ax.axhline(1.0, color="0.6", linestyle="--")
ax.set(xlabel=r"Observed velocity [km s$^{-1}$]", ylabel=r"$F/F_0$")
fig.tight_layout()
plt.show()
