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

lib.Line_Profile_Inflow.argtypes = [
    c_double_p, ctypes.c_size_t, ctypes.c_double, c_double_p,
    c_double_p, c_double_p, ctypes.c_size_t,
    c_double_p, c_double_p, c_double_p, ctypes.c_size_t,
    c_int_p, c_int_p, c_double_p, c_double_p,
    c_int_p, ctypes.c_size_t,
    ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
    ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
    ctypes.c_double, ctypes.c_double, ctypes.c_int, ctypes.c_int,
    ctypes.c_int, c_double_p,
]
lib.Line_Profile_Inflow.restype = None


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


def _salt_outflow(v_obs, lam_ref, background, flow_parameters, profile_parameters, profile_type):
    """Call the turbulent-outflow C implementation."""

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
    v_b = flow_parameters["v_b"]
    f_c = flow_parameters["f_c"]
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
    _require_equal_lengths(
        "absorption_parameters", abs_waves.size,
        abs_osc_strs=abs_osc, abs_ein=abs_ein,
    )

    em_waves = _arr(emission["em_waves"])
    emitted_waves = _arr(emission["emitted_waves"])
    em_osc = _arr(emission["em_osc_strs"])
    em_ein = _arr(emission["em_ein"])

    res = _arr(emission["res"], np.int32)
    fluor = _arr(emission["fluor"], np.int32)
    line_num = _arr(emission["line_num"], np.int32)
    p_r = _arr(emission["p_r"])
    p_f = _arr(emission["p_f"])
    _require_equal_lengths(
        "emission_parameters", em_waves.size,
        emitted_waves=emitted_waves, em_osc_strs=em_osc, em_ein=em_ein,
        res=res, fluor=fluor, p_r=p_r, p_f=p_f,
    )
    if line_num.size != abs_waves.size or np.any(line_num < 0):
        raise ValueError("outflow line_num must have one nonnegative count per absorption line")
    if int(line_num.sum()) != em_waves.size:
        raise ValueError("outflow line_num must sum to len(em_waves)")
    atomic_arrays = (abs_waves, abs_osc, abs_ein, em_waves, emitted_waves, em_osc, em_ein, p_r, p_f)
    if any(not np.all(np.isfinite(values)) for values in atomic_arrays):
        raise ValueError("outflow atomic inputs and probabilities must be finite")
    if np.any(abs_waves <= 0.0) or np.any(em_waves <= 0.0) or np.any(emitted_waves <= 0.0):
        raise ValueError("all outflow wavelengths must be positive")
    if np.any(abs_osc < 0.0) or np.any(em_osc < 0.0):
        raise ValueError("outflow oscillator strengths cannot be negative")
    if np.any(abs_ein < 0.0) or np.any(em_ein < 0.0):
        raise ValueError("Einstein A coefficients cannot be negative")
    if np.any((p_r < 0.0) | (p_r > 1.0)) or np.any((p_f < 0.0) | (p_f > 1.0)):
        raise ValueError("p_r and p_f must lie between 0 and 1")

    blend_flag = _arr(blending["blending"], np.int32)
    if blend_flag.size != em_waves.size:
        raise ValueError("blending must have one flag per outflow emission channel")
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
        f_c, k_dust, delta,
        int(APERTURE), int(OCCULTATION),
        SW,
        int(use_sobolev_wings),
        profile_method,

        ptype,
        out,
    )

    return out


def _require_equal_lengths(group_name, expected, **arrays):
    for field, values in arrays.items():
        if values.size != expected:
            raise ValueError(
                f"{group_name}[{field!r}] has length {values.size}; "
                f"expected {expected}"
            )


def _profile_code(profile_type):
    try:
        return {"absorption": 0, "emission": 1, "pcygni": 2}[profile_type]
    except KeyError as exc:
        raise ValueError(
            "profile_type must be 'absorption', 'emission', or 'pcygni'"
        ) from exc


def _validate_shared_inputs(v_obs, lam_ref, background, flow_parameters):
    v_obs = _arr(v_obs)
    background = _arr(background)
    if v_obs.ndim != 1 or background.ndim != 1:
        raise ValueError("v_obs and background must be one-dimensional")
    if v_obs.size == 0:
        raise ValueError("v_obs cannot be empty")
    if background.size != v_obs.size:
        raise ValueError("background and v_obs must have the same length")
    if not np.all(np.isfinite(v_obs)) or not np.all(np.isfinite(background)):
        raise ValueError("v_obs and background must contain only finite values")
    if np.any(np.diff(v_obs) <= 0.0):
        raise ValueError("v_obs must be strictly increasing")
    if not np.isfinite(lam_ref) or lam_ref <= 0.0:
        raise ValueError("lam_ref must be a positive finite wavelength")

    required = {"alpha", "psi", "gamma", "tau", "v_0", "v_w", "f_c", "k", "delta"}
    missing = sorted(required.difference(flow_parameters))
    if missing:
        raise KeyError(f"flow_parameters is missing: {', '.join(missing)}")

    values = {name: float(flow_parameters[name]) for name in required}
    if not all(np.isfinite(value) for value in values.values()):
        raise ValueError("all flow parameters must be finite")
    if not 0.0 <= values["alpha"] <= np.pi / 2.0:
        raise ValueError("alpha must lie between 0 and pi/2 radians")
    if not 0.0 <= values["psi"] <= np.pi / 2.0:
        raise ValueError("psi must lie between 0 and pi/2 radians")
    if values["gamma"] <= 0.0:
        raise ValueError("gamma must be positive")
    if values["tau"] < 0.0:
        raise ValueError("tau cannot be negative")
    if values["v_0"] <= 0.0 or values["v_w"] <= values["v_0"]:
        raise ValueError("require 0 < v_0 < v_w")
    if not 0.0 <= values["f_c"] <= 1.0:
        raise ValueError("f_c must lie between 0 and 1")
    if values["k"] < 0.0:
        raise ValueError("k cannot be negative")
    if values["delta"] <= 0.0:
        raise ValueError("delta must be positive")
    return v_obs, background


def _salt_inflow(v_obs, lam_ref, background, flow_parameters, profile_parameters, profile_type):
    """Call the inflow C implementation."""
    if "v_b" in flow_parameters:
        raise ValueError("v_b applies only to model_type='outflow'")

    try:
        absorption = profile_parameters["absorption_parameters"]
        emission = profile_parameters["emission_parameters"]
        observing = profile_parameters["observing_parameters"]
        misc = profile_parameters["miscellaneous_parameters"]
    except KeyError as exc:
        raise KeyError(
            "inflow profile_parameters must use the shared nested layout: "
            "absorption_parameters, emission_parameters, "
            "observing_parameters, and miscellaneous_parameters"
        ) from exc

    abs_waves = _arr(absorption["abs_waves"])
    abs_osc = _arr(absorption["abs_osc_strs"])
    _require_equal_lengths(
        "absorption_parameters", abs_waves.size, abs_osc_strs=abs_osc
    )

    em_waves = _arr(emission["em_waves"])
    em_osc = _arr(emission["em_osc_strs"])
    emitted_waves = _arr(emission["emitted_waves"])
    resonance = _arr(emission["res"], np.int32)
    fluorescence = _arr(emission["fluor"], np.int32)
    p_r = _arr(emission["p_r"])
    p_f = _arr(emission["p_f"])
    line_num = _arr(emission["line_num"], np.int32)
    _require_equal_lengths(
        "emission_parameters", em_waves.size,
        em_osc_strs=em_osc, emitted_waves=emitted_waves,
        res=resonance, fluor=fluorescence, p_r=p_r, p_f=p_f,
    )
    if line_num.size != abs_waves.size or np.any(line_num < 0):
        raise ValueError(
            "inflow line_num must have one nonnegative count per absorption line"
        )
    if int(line_num.sum()) != em_waves.size:
        raise ValueError("inflow line_num must sum to len(em_waves)")

    atomic_arrays = (abs_waves, abs_osc, em_waves, em_osc, emitted_waves, p_r, p_f)
    if any(not np.all(np.isfinite(values)) for values in atomic_arrays):
        raise ValueError("atomic inputs and branching probabilities must be finite")
    if np.any(abs_waves <= 0.0) or np.any(em_waves <= 0.0) or np.any(emitted_waves <= 0.0):
        raise ValueError("all wavelengths must be positive")
    if np.any(abs_osc < 0.0) or np.any(em_osc < 0.0):
        raise ValueError("oscillator strengths cannot be negative")
    if np.any((p_r < 0.0) | (p_r > 1.0)) or np.any((p_f < 0.0) | (p_f > 1.0)):
        raise ValueError("p_r and p_f must lie between 0 and 1")

    alpha = float(flow_parameters["alpha"])
    psi = float(flow_parameters["psi"])
    gamma = float(flow_parameters["gamma"])
    tau = float(flow_parameters["tau"])
    v_0 = float(flow_parameters["v_0"])
    v_w = float(flow_parameters["v_w"])
    f_c = float(flow_parameters["f_c"])
    k_dust = float(flow_parameters["k"])
    delta = float(flow_parameters["delta"])
    try:
        v_ap = float(observing["v_ap"])
    except KeyError as exc:
        raise KeyError(
            "inflow observing_parameters requires 'v_ap'"
        ) from exc
    aperture = int(bool(observing.get("APERTURE", True)))
    occultation = int(bool(misc.get("OCCULTATION", True)))

    if "v_obs" in observing:
        observing_v_obs = _arr(observing["v_obs"])
        if observing_v_obs.shape != v_obs.shape or not np.array_equal(
            observing_v_obs, v_obs
        ):
            raise ValueError(
                "observing_parameters['v_obs'] must match the v_obs argument"
            )
    if "lam_ref" in misc and float(misc["lam_ref"]) != float(lam_ref):
        raise ValueError(
            "miscellaneous_parameters['lam_ref'] must match the lam_ref argument"
        )

    blending = profile_parameters.get("blending_parameters")
    if blending is not None and any(bool(flag) for flag in blending.get("blending", [])):
        raise ValueError("transition blending is not implemented for model_type='inflow'")
    unsupported_numerics = {"Sobolev", "SW", "profile_method"}.intersection(misc)
    if unsupported_numerics:
        names = ", ".join(sorted(unsupported_numerics))
        raise ValueError(f"{names} applies only to model_type='outflow'")
    if not np.isfinite(v_ap) or v_ap < 0.0:
        raise ValueError("inflow v_ap must be finite and nonnegative")
    if aperture and v_ap >= v_w:
        raise ValueError("with APERTURE=True, inflow v_ap must be less than v_w")

    output = np.empty_like(v_obs)
    lib.Line_Profile_Inflow(
        v_obs, v_obs.size, float(lam_ref), background,
        abs_waves, abs_osc, abs_waves.size,
        em_waves, emitted_waves, em_osc, em_waves.size,
        resonance, fluorescence, p_r, p_f,
        line_num, line_num.size,
        alpha, psi, gamma, tau,
        v_0, v_w, v_ap, f_c, k_dust, delta,
        aperture, occultation, _profile_code(profile_type), output,
    )
    if not np.all(np.isfinite(output)):
        raise RuntimeError("the C inflow model returned non-finite values")
    return output


def salt(
    v_obs,
    lam_ref,
    background,
    flow_parameters,
    profile_parameters,
    profile_type,
    model_type,
):
    """Return a continuum-normalized SALT spectrum.

    ``model_type`` is required and must be ``"outflow"`` or ``"inflow"``.
    Wavelengths are in Angstrom, velocities in km/s, and angles in radians.
    """
    if model_type not in {"outflow", "inflow"}:
        raise ValueError("model_type must be 'outflow' or 'inflow'")
    v_obs, background = _validate_shared_inputs(
        v_obs, lam_ref, background, flow_parameters
    )
    _profile_code(profile_type)

    if model_type == "outflow":
        if "v_ap" in flow_parameters:
            raise ValueError(
                "for model_type='outflow', place v_ap in "
                "profile_parameters['observing_parameters'], not flow_parameters"
            )
        if "v_b" not in flow_parameters:
            raise KeyError("outflow flow_parameters requires 'v_b'")
        v_b = float(flow_parameters["v_b"])
        if not np.isfinite(v_b) or v_b < 0.0:
            raise ValueError("outflow v_b must be finite and nonnegative")
        try:
            observing = profile_parameters["observing_parameters"]
            v_ap = float(observing["v_ap"])
        except KeyError as exc:
            raise KeyError(
                "outflow profile_parameters requires observing_parameters['v_ap']"
            ) from exc
        if not np.isfinite(v_ap) or not 0.0 < v_ap <= float(flow_parameters["v_w"]):
            raise ValueError("outflow v_ap must satisfy 0 < v_ap <= v_w")
        return _salt_outflow(
            v_obs, lam_ref, background, flow_parameters,
            profile_parameters, profile_type,
        )
    if "v_ap" in flow_parameters:
        raise ValueError(
            "for model_type='inflow', place v_ap in "
            "profile_parameters['observing_parameters'], not flow_parameters"
        )
    return _salt_inflow(
        v_obs, lam_ref, background, flow_parameters,
        profile_parameters, profile_type,
    )


__all__ = ["salt"]
