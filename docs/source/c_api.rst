C API
=====

The declarations are in :code:`include/salt.h`.

``Line_Profile``
----------------

Computes the turbulent-outflow profile. It accepts the velocity grid,
background, absorption and emission channels, flattened transition-blending
buffers, flow parameters, observing options, Voigt options, and a caller-owned
output array.

``Line_Profile_Inflow``
-----------------------

Computes the Sobolev inflow profile. It has a separate signature because the
inflow model does not currently use Einstein A coefficients, turbulent
broadening, Voigt selection, or transition blending.

Public declarations
-------------------

.. literalinclude:: ../../include/salt.h
   :language: c
   :linenos:

The Python dispatcher is recommended for ordinary use because it validates
the inputs and constructs the flattened arrays required by the C interface.
