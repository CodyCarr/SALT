Installation
============

Dependencies
------------

C SALT requires a C11 compiler, GSL, libcerf, OpenMP, Python 3, and NumPy.
Matplotlib is needed only for the plotting examples.

On macOS with Homebrew:

.. code-block:: console

   brew install gsl libcerf libomp
   python -m pip install numpy matplotlib

On Debian or Ubuntu:

.. code-block:: console

   sudo apt update
   sudo apt install build-essential libgsl-dev libcerf-dev libomp-dev python3-dev
   python3 -m pip install numpy matplotlib

Download and compile
--------------------

.. code-block:: console

   git clone https://github.com/CodyCarr/SALT.git
   cd SALT
   make

The build creates :code:`libsalt.dylib` on macOS or :code:`libsalt.so` on
Linux. Run :code:`make debug` for an unoptimized build with debug symbols and
:code:`make clean` to remove generated files. The compiler may be overridden,
for example with :code:`make CC=gcc`.

Verify the installation
-----------------------

Run the two self-contained examples from the repository root:

.. code-block:: console

   python python/outflow_example.py
   python python/inflow_example.py

Small platform-dependent numerical differences are acceptable. The reference
flux ranges are approximately:

.. code-block:: text

   Flux range: 0.063515 to 2.584356  # outflow
   Flux range: 0.020925 to 1.088382  # inflow
