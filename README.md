# C SALT

C SALT is a C implementation of the Semi-Analytical Line Transfer (SALT)
model for continuum absorption and resonant/fluorescent re-emission by
spherical and biconical galactic winds. The model includes thermal and
microturbulent line broadening, dust attenuation, finite apertures, source
occultation, and attenuation by overlapping transitions.

The public C entry point is `Line_Profile` in `include/salt.h`. Most users
should use the Python interface, `salt()` in `python/run_salt.py`.

## Physical basis

This implementation is based on:

- Carr et al. (2023), *Testing SALT Approximations with Numerical Radiation
  Transfer Code. I. Validity and Applicability*, ApJ, 952, 88,
  <https://doi.org/10.3847/1538-4357/acd331>.
- Carr et al., *Testing SALT Approximations with Numerical Radiative Transfer
  Code. II. Thermal and Microturbulent Line Broadening* (manuscript associated
  with this release).

## Dependencies

Required:

- A C11 compiler
- GNU Scientific Library (GSL)
- libcerf
- OpenMP
- Python 3 and NumPy for the Python interface

Matplotlib is required only by the plotting example.

### macOS with Homebrew

```sh
brew install gsl libcerf libomp
python -m pip install numpy matplotlib
```

The Makefile obtains the active Homebrew prefix automatically, supporting both
Apple Silicon and Intel Homebrew installations.

### Debian or Ubuntu Linux

```sh
sudo apt update
sudo apt install build-essential libgsl-dev libcerf-dev libomp-dev python3-dev
python3 -m pip install numpy matplotlib
```

The default Linux compiler can be overridden when necessary, for example with
`make CC=gcc`.

## Build

From the repository root, run `make`. This produces `libsalt.dylib` on macOS
or `libsalt.so` on Linux. Use `make debug` for an unoptimized build with debug
symbols and `make clean` to remove generated files.

## Quick start

Run the self-contained example from the repository root:

```sh
python python/example.py
```

It calculates and plots a single-line P-Cygni profile. A successful run prints
a finite flux range close to:

```text
Flux range: 0.468309 to 2.463790
```

Small numerical differences across compilers and platforms are acceptable.

The essential call is:

```python
from run_salt import salt

spectrum = salt(
    v_obs=v_obs,
    lam_ref=lam_ref,
    background=background,
    flow_parameters=flow_parameters,
    profile_parameters=profile_parameters,
    profile_type="pcygni",
)
```

See `python/example.py` for complete parameter dictionaries.

## Parameter conventions

All wavelengths are in Angstrom, velocities are in km s\(^{-1}\), and angles
are in radians.

### Wind parameters

| Key | Meaning |
| --- | --- |
| `alpha` | Bicone half-opening angle |
| `psi` | Angle between the bicone axis and line of sight |
| `gamma` | Power-law index of the velocity field |
| `tau` | SALT optical-depth normalization |
| `v_0` | Launch velocity |
| `v_w` | Terminal wind velocity |
| `v_b` | Doppler parameter for thermal/microturbulent broadening |
| `f_c` | Wind covering fraction |
| `k` | Dust-opacity normalization |
| `delta` | Power-law index of the density field |

`v_b` is the Doppler parameter, not the one-dimensional Gaussian standard
deviation: \(\sigma_v=v_b/\sqrt{2}\). Some internal legacy variables are
named `f_holes`; they represent the covering fraction supplied as `f_c`.

### Observing and numerical parameters

| Key | Meaning |
| --- | --- |
| `v_obs` | Observed-velocity sampling grid |
| `lam_ref` | Reference wavelength defining zero observed velocity |
| `background` | Incident continuum sampled on `v_obs` |
| `v_ap` | Aperture velocity corresponding to the projected aperture radius |
| `APERTURE` | Enable the finite-aperture calculation |
| `OCCULTATION` | Enable source occultation of receding emission |
| `profile_type` | `"absorption"`, `"emission"`, or `"pcygni"` |
| `profile_method` | `"wofz"` or `"colt"` Voigt evaluation |
| `Sobolev` | Select hybrid Sobolev/Voigt mode when `True` |
| `SW` | Half-width of the central Voigt region in km s\(^{-1}\) |

With `Sobolev=True`, C SALT evaluates the turbulent Voigt calculation for
\(|v_{\rm obs}|<SW\) and uses the faster Sobolev-limit geometry outside this
interval. This is a **hybrid Sobolev/Voigt mode**, not a purely Sobolev
calculation.

With `Sobolev=False`, `SW` is not read and may be omitted. The Voigt
calculation is evaluated at every observed velocity and requires `v_b > 0`.

`profile_method="wofz"` evaluates the Faddeeva function using libcerf.
`profile_method="colt"` uses the continued-fraction approximation described
by Smith et al. (2015, Appendix A1).

### Atomic and branching parameters

| Key | Meaning |
| --- | --- |
| `abs_waves`, `em_waves` | Absorbing transition wavelengths |
| `emitted_waves` | Wavelengths of emitted channels |
| `abs_osc_strs`, `em_osc_strs` | Absorption oscillator strengths |
| `abs_ein`, `em_ein` | Einstein A coefficients in s\(^{-1}\) |
| `res`, `fluor` | Resonant and fluorescent channel flags |
| `p_r`, `p_f` | Resonant-survival and fluorescent probabilities |
| `line_num` | Emission-channel count for each absorption line |
| `blending` | Enable attenuation/re-emission by neighboring transitions |

The Python wrapper flattens the ragged blending arrays. Each emission channel
must have an entry; see `python/example.py` for the required placeholders when
blending is disabled.

## Numerical resolution

Compile-time grid sizes are centralized in `include/salt_grid_config.h`.
Increasing them generally improves numerical resolution at the cost of run
time. Any change should be followed by convergence testing over the intended
parameter range.

## Benchmarking

The first call can include shared-library, OpenMP, and cache initialization.
For performance representative of fitting or MCMC use, make one untimed
warm-up call and report the median of several subsequent calls. The quick-start
example demonstrates this procedure. Include the first call only when
measuring single-use latency.

## Model scope

The emission calculation uses the SALT shell-based escape-probability
formalism. It is not a general multiple-scattering Monte Carlo solver and does
not model unrestricted spatial and frequency diffusion at very high optical
depth. Consult the associated papers for the assumptions and validated
parameter regime.

## Citation

Please cite the model papers when using this code:

```bibtex
@article{Carr2023,
  author  = {Carr, Cody and Michel-Dansac, Leo and Blaizot, Jeremy and
             Scarlata, Claudia and Henry, Alaina and Verhamme, Anne},
  title   = {Testing SALT Approximations with Numerical Radiation Transfer
             Code. I. Validity and Applicability},
  journal = {The Astrophysical Journal},
  volume  = {952},
  number  = {1},
  pages   = {88},
  year    = {2023},
  doi     = {10.3847/1538-4357/acd331}
}

@unpublished{Carr2026,
  author = {Carr, Cody A. and Cen, Renyue and Michel-Dansac, Leo and
            Scarlata, Claudia and Henry, Alaina},
  title  = {Testing SALT Approximations with Numerical Radiative Transfer
            Code. II. Thermal and Microturbulent Line Broadening},
  year   = {2026},
  note   = {Manuscript associated with this software release}
}
```

Replace the second entry with the final journal citation after publication.

## Tested configuration

The current release was built and exercised on macOS 26.6.2 with Apple Clang
21.0.0, Python 3.13.9, NumPy 2.3.5, GSL 2.8, libcerf 3.3, and libomp 22.1.0.
Other recent C11/Python environments should work but have not yet been added
to the tested-platform list.

## Support and contributions

For scientific or implementation questions, contact Cody A. Carr at
`codycarr24@gmail.com`. Bug reports should include the platform, compiler and
dependency versions, a minimal parameter dictionary, and the expected and
actual behavior. Contributions should preserve the public API where practical
and include a numerical comparison against an existing reference case.

## License

C SALT is distributed under the BSD 3-Clause License. See `LICENSE.md` for the
full terms. The license permits use, modification, and redistribution in source
or binary form while requiring preservation of the copyright and license
notices and prohibiting the use of the copyright holder's or contributors'
names to endorse derived products without written permission.
