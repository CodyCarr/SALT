Parameter reference
===================

Conventions
-----------

Wavelengths are in Angstrom, velocities are in km/s, and angles are in
radians. :code:`v_obs` must be a finite, strictly increasing one-dimensional
array. :code:`background` must be finite and have the same length.

Shared flow parameters
----------------------

The :code:`flow_parameters` dictionary contains:

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Key
     - Meaning
   * - :code:`alpha`
     - Bicone half-opening angle in :math:`[0,\pi/2]`.
   * - :code:`psi`
     - Angle between the bicone axis and line of sight in :math:`[0,\pi/2]`.
   * - :code:`gamma`
     - Positive power-law index of the velocity field.
   * - :code:`tau`
     - Nonnegative SALT optical-depth normalization.
   * - :code:`v_0`
     - Positive launch velocity.
   * - :code:`v_w`
     - Terminal velocity, with :math:`v_w>v_0`.
   * - :code:`f_c`
     - Covering fraction in :math:`[0,1]`.
   * - :code:`k`
     - Nonnegative dust-opacity normalization.
   * - :code:`delta`
     - Positive power-law index of the density field.
   * - :code:`v_b`
     - Outflow only: nonnegative Doppler parameter. It is not the Gaussian
       standard deviation; :math:`\sigma_v=v_b/\sqrt{2}`.

Nested profile dictionaries
---------------------------

:code:`profile_parameters` contains
:code:`absorption_parameters`, :code:`emission_parameters`,
:code:`observing_parameters`, and :code:`miscellaneous_parameters`. The
outflow model also requires :code:`blending_parameters`.

Atomic and branching keys
^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Key
     - Meaning
   * - :code:`abs_waves`, :code:`em_waves`
     - Absorbing or parent-transition wavelengths.
   * - :code:`emitted_waves`
     - Wavelength of each emitted channel.
   * - :code:`abs_osc_strs`, :code:`em_osc_strs`
     - Oscillator strengths corresponding to the parent transitions.
   * - :code:`abs_ein`, :code:`em_ein`
     - Outflow-only Einstein A coefficients in s\ :sup:`-1`.
   * - :code:`res`, :code:`fluor`
     - Boolean resonant and fluorescent flags for every emission channel.
   * - :code:`p_r`, :code:`p_f`
     - Resonant-survival and fluorescent branching probabilities.
   * - :code:`line_num`
     - Number of emission channels associated with each absorption line. Its
       sum must equal the length of :code:`em_waves`.

Observing and miscellaneous keys
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Key
     - Location
     - Meaning
   * - :code:`v_ap`
     - observing
     - Aperture velocity corresponding to the projected aperture radius.
   * - :code:`APERTURE`
     - observing
     - Enable the finite-aperture calculation.
   * - :code:`OCCULTATION`
     - miscellaneous
     - Enable source occultation of receding emission.
   * - :code:`Sobolev`
     - miscellaneous
     - Outflow only: enable hybrid Sobolev/Voigt mode.
   * - :code:`SW`
     - miscellaneous
     - Outflow only: half-width of the central Voigt region in km/s.
   * - :code:`profile_method`
     - miscellaneous
     - Outflow only: :code:`"wofz"` or :code:`"colt"`.

Outflow blending
----------------

The outflow wrapper packs ragged blending arrays into fixed-size C buffers.
Each emission channel needs an entry in :code:`blended_waves`,
:code:`blended_osc_strs`, :code:`blended_abs_ein`, :code:`blended_fluor`,
:code:`blended_p_r`, :code:`blended_p_f`,
:code:`blended_flour_waves`, and :code:`blending`. Disabled channels use the
zero-valued placeholders shown in :doc:`examples`.

The spelling :code:`blended_flour_waves` is retained by the public API for
compatibility.
