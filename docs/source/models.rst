Choosing a model
================

The :code:`model_type` argument is required. This prevents an inflow parameter
set from being silently evaluated with the outflow equations, or vice versa.

Turbulent outflow
-----------------

Use :code:`model_type="outflow"` for an expanding spherical or biconical
flow. This model supports a Doppler parameter :math:`v_b`, Voigt absorption,
resonant and fluorescent emission, dust, finite apertures, occultation, and
attenuation or re-emission by neighboring transitions.

The outflow model can use either the libcerf Faddeeva function
(:code:`profile_method="wofz"`) or the COLT continued-fraction approximation
(:code:`profile_method="colt"`). See :doc:`numerical` for the hybrid
Sobolev/Voigt switch.

Inflow
------

Use :code:`model_type="inflow"` for the contracting-flow model described by
Carr & Scarlata (2022). Inflow speeds are supplied as positive magnitudes; the
C dispatcher applies the internal sign convention.

The current inflow model uses the Sobolev approximation. It does not accept
:code:`v_b`, :code:`Sobolev`, :code:`SW`, or :code:`profile_method`, and it
does not implement transition blending. It does support dust, finite
apertures, source occultation, and resonant or fluorescent emission.

Profile components
------------------

The :code:`profile_type` argument accepts:

* :code:`"absorption"` for continuum absorption only;
* :code:`"emission"` for scattered emission only; or
* :code:`"pcygni"` for the combined profile.

Scope
-----

The emission calculation uses the SALT shell-based escape-probability
formalism. It is not a general Monte Carlo solver and does not model
unrestricted spatial and frequency diffusion at very high optical depth.
Consult the model papers for the assumptions and validated regimes.
