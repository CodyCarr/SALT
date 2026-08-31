# SPDX-License-Identifier: BSD-3-Clause
"""Minimal, self-contained C SALT turbulent-outflow example."""

import time

import matplotlib.pyplot as plt
import numpy as np

from run_salt import salt


# Center the velocity grid on the red member of the Si II doublet.  This range
# also contains Si II 1190 and the associated Si II* fluorescent channels.
v_obs = np.linspace(-2000.0, 2000.0, 1000)
lam_ref = 1193.28
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
    "abs_waves": [1190.42, 1193.28],
    "abs_osc_strs": [0.277, 0.575],
    "abs_ein": [6.53e8, 2.69e9],
}

emission_parameters = {
    "em_waves": [1190.42, 1190.42, 1193.28, 1193.28],
    "emitted_waves": [1190.42, 1194.50, 1193.28, 1197.39],
    "em_osc_strs": [0.277, 0.277, 0.575, 0.575],
    "em_ein": [6.53e8, 6.53e8, 2.69e9, 2.69e9],
    "res": [True, False, True, False],
    "fluor": [False, True, False, True],
    "p_r": [0.1592, 0.1592, 0.6577, 0.6577],
    "p_f": [0.8408, 0.8408, 0.3423, 0.3423],
    "line_num": [2, 2],
}

# without blending
# Each emission channel needs a blending entry. The flag is False here, so
# the zero-valued placeholders are packed but never used by the C kernel.
blending_parameters = {
    "blended_waves": [[0.0], [0.0], [0.0], [0.0]],
    "blended_osc_strs": [[0.0], [0.0], [0.0], [0.0]],
    "blended_abs_ein": [[0.0], [0.0], [0.0], [0.0]],
    "blended_fluor": [[False], [False], [False], [False]],
    "blended_p_r": [[0.0], [0.0], [0.0], [0.0]],
    "blended_p_f": [[[0.0]], [[0.0]], [[0.0]], [[0.0]]],
    "blended_flour_waves": [[[0.0]], [[0.0]], [[0.0]], [[0.0]]],
    "blending": [False, False, False, False],
}

# with blending, typical only needed at high column densities log N/cm^-2 > 17
# blending_parameters = {
#     "blended_waves": [[1193.28], [1193.28], [0.0], [0.0]],
#     "blended_osc_strs": [[0.575], [0.575], [0.0], [0.0]],
#     "blended_abs_ein": [[2.69e9], [2.69e9], [0.0], [0.0]],
#     "blended_fluor": [[True], [True], [False], [False]],
#     "blended_p_r": [[0.6577], [0.6577], [0.0], [0.0]],
#     "blended_p_f": [[[0.3423]], [[0.3423]], [[0.0]], [[0.0]]],
#     "blended_flour_waves": [[[1197.39]], [[1197.39]], [[0.0]], [[0.0]]],
#     "blending": [True, True, False, False],
# }


observing_parameters = {"APERTURE": True, "v_ap": 500.0, "v_obs": v_obs}
miscellaneous_parameters = {
    "OCCULTATION": False,
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
    "model_type": "outflow",
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
line_waves = np.array([1190.42, 1194.50, 1193.28, 1197.39])
zeros = c_kms * (line_waves / lam_ref - 1.0)

fig, ax = plt.subplots(figsize=(7, 5))

ax.vlines(zeros[[0, 2]], ymin, ymax, colors="b", linestyles="-", alpha=0.6)
ax.vlines(zeros[[1, 3]], ymin, ymax, colors="b", linestyles="--", alpha=0.6)
ax.plot(v_obs, spectrum, color="tab:red", linewidth=2)

ax.axhline(1.0, color="0.6", linestyle="--")
ax.set(xlabel=r"Observed velocity [km s$^{-1}$]", ylabel=r"$F/F_0$")
ax.set_xlim(-1500,1500)
ax.set_ylim(ymin,ymax)
ax.grid()
fig.tight_layout()
plt.show()
