# SPDX-License-Identifier: BSD-3-Clause

import ctypes
import os
import numpy as np
from numpy.ctypeslib import ndpointer

# The flattened limits below must match the private C storage in
# SALT2026_LineProfile.c and SALT2026_Emission.c.
MAX_BLENDS = 32
MAX_FLUOR = 16

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_candidates = [os.path.join(_root, "libsalt.dylib"), os.path.join(_root, "libsalt.so")]
_libpath = next((path for path in _candidates if os.path.exists(path)), None)
if _libpath is None:
    raise FileNotFoundError("SALT shared library not found; run 'make' in the repository root")
lib = ctypes.CDLL(_libpath)

c_double_p = ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
c_int_p = ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")

lib.Line_Profile.argtypes = [
    c_double_p, ctypes.c_int,
    ctypes.c_double,
    c_double_p,

    c_double_p, c_double_p, c_double_p, ctypes.c_int,

    c_double_p, c_double_p, c_double_p, c_double_p, ctypes.c_int,

    c_int_p, c_int_p, c_int_p,
    c_double_p, c_double_p,

    c_double_p, c_double_p, c_double_p, c_int_p,
    c_double_p, c_double_p, c_double_p,
    c_int_p, c_int_p,

    c_int_p, ctypes.c_int,

    ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
    ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
    ctypes.c_double, ctypes.c_double, ctypes.c_double,
    ctypes.c_int, ctypes.c_int,
    ctypes.c_double,
    ctypes.c_int,
    ctypes.c_int,

    ctypes.c_int,
    c_double_p,
]
lib.Line_Profile.restype = None


def _arr(x, dtype=np.float64):
    return np.ascontiguousarray(x, dtype=dtype)


def _pack_blends(bp, n_em):
    """Pack ragged Python blending data into fixed-size C buffers."""
    waves = np.zeros((n_em, MAX_BLENDS), dtype=np.float64)
    osc = np.zeros_like(waves)
    ein = np.zeros_like(waves)
    fluor = np.zeros((n_em, MAX_BLENDS), dtype=np.int32)
    pr = np.zeros_like(waves)
    pf = np.zeros((n_em, MAX_BLENDS, MAX_FLUOR), dtype=np.float64)
    fluor_waves = np.zeros_like(pf)
    n_fluor = np.zeros((n_em, MAX_BLENDS), dtype=np.int32)
    n_blends = np.zeros(n_em, dtype=np.int32)

    bw = bp["blended_waves"]
    bo = bp["blended_osc_strs"]
    be = bp["blended_abs_ein"]
    bf = bp["blended_fluor"]
    bpr = bp["blended_p_r"]
    bpf = bp["blended_p_f"]
    bfw = bp["blended_flour_waves"]

    for i in range(n_em):
        nbi = min(len(bw[i]), MAX_BLENDS)
        n_blends[i] = nbi
        for j in range(nbi):
            waves[i, j] = bw[i][j]
            osc[i, j] = bo[i][j]
            ein[i, j] = be[i][j]
            fluor[i, j] = int(bf[i][j])
            pr[i, j] = bpr[i][j]

            nf = min(len(bfw[i][j]), MAX_FLUOR)
            n_fluor[i, j] = nf
            for k in range(nf):
                pf[i, j, k] = bpf[i][j][k]
                fluor_waves[i, j, k] = bfw[i][j][k]

    return (
        np.ascontiguousarray(waves.ravel()),
        np.ascontiguousarray(osc.ravel()),
        np.ascontiguousarray(ein.ravel()),
        np.ascontiguousarray(fluor.ravel(), dtype=np.int32),
        np.ascontiguousarray(pr.ravel()),
        np.ascontiguousarray(pf.ravel()),
        np.ascontiguousarray(fluor_waves.ravel()),
        np.ascontiguousarray(n_fluor.ravel(), dtype=np.int32),
        n_blends,
    )


def salt(v_obs, lam_ref, background, flow_parameters, profile_parameters, profile_type):
    """Return a continuum-normalized SALT spectrum on ``v_obs``.

    Wavelengths are in Angstrom, velocities in km/s, and angular parameters
    in radians. ``profile_type`` is ``"absorption"``, ``"emission"``, or
    ``"pcygni"``. The nested parameter dictionaries follow the worked example
    in ``example.py``.
    """
    absorption = profile_parameters["absorption_parameters"]
    emission = profile_parameters["emission_parameters"]
    blending = profile_parameters["blending_parameters"]
    observing = profile_parameters["observing_parameters"]
    misc = profile_parameters["miscellaneous_parameters"]

    alpha = flow_parameters["alpha"]
    psi = flow_parameters["psi"]
    gamma = flow_parameters["gamma"]
    tau = flow_parameters["tau"]
    v_0 = flow_parameters["v_0"]
    v_w = flow_parameters["v_w"]
    # ``v_th`` was the historical name for the Doppler parameter. Accept it
    # temporarily so existing scripts remain usable, but expose ``v_b`` in all
    # new examples and documentation.
    try:
        v_b = flow_parameters["v_b"]
    except KeyError:
        v_b = flow_parameters["v_th"]
    f_holes = flow_parameters["f_c"]
    k_dust = flow_parameters["k"]
    delta = flow_parameters["delta"]

    v_ap = observing["v_ap"]
    APERTURE = observing["APERTURE"]
    OCCULTATION = misc["OCCULTATION"]
    use_sobolev_wings = bool(misc.get("Sobolev", True))
    # SW is intentionally not accessed in full-Voigt mode. In hybrid mode it
    # is supplied in km/s and normalized by |v_0| inside the C kernel.
    SW = float(misc["SW"]) if use_sobolev_wings else 0.0
    if not use_sobolev_wings and v_b <= 0.0:
        raise ValueError("Sobolev=False requires v_b > 0 km/s")
    profile_method_name = misc.get("profile_method", "wofz")
    try:
        profile_method = {"wofz": 0, "colt": 1}[profile_method_name]
    except KeyError as exc:
        raise ValueError("profile_method must be 'wofz' or 'colt'") from exc

    abs_waves = _arr(absorption["abs_waves"])
    abs_osc = _arr(absorption["abs_osc_strs"])
    abs_ein = _arr(absorption["abs_ein"])

    em_waves = _arr(emission["em_waves"])
    emitted_waves = _arr(emission["emitted_waves"])
    em_osc = _arr(emission["em_osc_strs"])
    em_ein = _arr(emission["em_ein"])

    res = _arr(emission["res"], np.int32)
    fluor = _arr(emission["fluor"], np.int32)
    line_num = _arr(emission["line_num"], np.int32)
    p_r = _arr(emission["p_r"])
    p_f = _arr(emission["p_f"])

    blend_flag = _arr(blending["blending"], np.int32)
    (
        blended_waves,
        blended_osc,
        blended_ein,
        blended_fluor,
        blended_p_r,
        blended_p_f,
        blended_fluor_waves,
        n_fluor_each,
        n_blends_each,
    ) = _pack_blends(blending, em_waves.size)

    ptype = {"absorption": 0, "emission": 1, "pcygni": 2}[profile_type]

    v_obs = _arr(v_obs)
    background = _arr(background)
    out = np.zeros_like(v_obs)

    lib.Line_Profile(
        v_obs, v_obs.size,
        float(lam_ref),
        background,

        abs_waves, abs_osc, abs_ein, abs_waves.size,

        em_waves, emitted_waves, em_osc, em_ein, em_waves.size,

        res, fluor, blend_flag,
        p_r, p_f,

        blended_waves,
        blended_osc,
        blended_ein,
        blended_fluor,
        blended_p_r,
        blended_p_f,
        blended_fluor_waves,
        n_fluor_each,
        n_blends_each,

        line_num, line_num.size,

        alpha, psi, gamma, tau,
        v_0, v_w, v_ap, v_b,
        f_holes, k_dust, delta,
        int(APERTURE), int(OCCULTATION),
        SW,
        int(use_sobolev_wings),
        profile_method,

        ptype,
        out,
    )

    return out
