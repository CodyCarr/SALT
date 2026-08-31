C SALT
======

C SALT is a C implementation of the Semi-Analytical Line Transfer framework
for continuum absorption and resonant or fluorescent re-emission by spherical
and biconical galactic flows. It provides separate turbulent-outflow and
inflow models through one Python interface.

The outflow implementation includes thermal and microturbulent broadening,
dust, finite apertures, occultation, and attenuation by overlapping
transitions. The current inflow implementation uses the Sobolev approximation
and does not include turbulent broadening or transition blending.

Most users should call :code:`salt()` from :code:`python/run_salt.py` and set
:code:`model_type` explicitly to :code:`"outflow"` or :code:`"inflow"`.

.. code-block:: python

   from run_salt import salt

   spectrum = salt(
       v_obs=v_obs,
       lam_ref=lam_ref,
       background=background,
       flow_parameters=flow_parameters,
       profile_parameters=profile_parameters,
       profile_type="pcygni",
       model_type="outflow",
   )

.. toctree::
   :maxdepth: 2
   :caption: Getting started

   installation
   models
   examples

.. toctree::
   :maxdepth: 2
   :caption: Reference

   parameters
   python_api
   c_api
   numerical

.. toctree::
   :maxdepth: 1
   :caption: Resources

   citations

Support
-------

For scientific or implementation questions, contact Cody A. Carr at
:code:`codycarr24@gmail.com`. Bug reports should include the platform,
compiler and dependency versions, a minimal parameter dictionary, and the
expected and actual behavior.
