Python API
==========

The public Python entry point is:

.. code-block:: python

   salt(
       v_obs,
       lam_ref,
       background,
       flow_parameters,
       profile_parameters,
       profile_type,
       model_type,
   )

Arguments
---------

``v_obs``
   Strictly increasing observed-velocity grid in km/s.

``lam_ref``
   Positive reference wavelength in Angstrom defining zero observed velocity.

``background``
   Incident continuum sampled on :code:`v_obs`.

``flow_parameters``
   Dictionary containing the shared physical parameters and, for outflows,
   :code:`v_b`.

``profile_parameters``
   Nested dictionaries containing atomic, observing, and numerical options.

``profile_type``
   One of :code:`"absorption"`, :code:`"emission"`, or :code:`"pcygni"`.

``model_type``
   Required selector: :code:`"outflow"` or :code:`"inflow"`.

Returns
-------

A NumPy array containing the continuum-normalized line profile sampled on
:code:`v_obs`.

Input validation
----------------

The wrapper validates array shapes, finite values, physical bounds, atomic
array lengths, branching probabilities, and model-specific options before
calling C. This is why :code:`model_type` has no default and unsupported
outflow options raise errors in the inflow model.
