Changelog
=========

v3.2.0
------

Added
~~~~~

-  New ``CompositeModel`` class that combines modular linear models and
   behaves like a simple ``Model``.

Changed
~~~~~~~

-  Fully reworked ``NoiseWaves`` class to use the ``CompositeModel``
   class.

v3.1.1
------

Changed
~~~~~~~

-  New ``model_filter`` function that ``xrfi_model`` calls and can be
   called by other more general functions.

Added
~~~~~

-  New tests of ``ModelFilterInfo`` and ``ModelFilterInfoContainer``

v3.1.0
------

Added
~~~~~

-  New semi-rigid S-parameters file from 2017.

v3.0.0
------

Fixed
~~~~~

-  Tests were slow because they were using the wrong default for getting
   the standard deviation in ``xrfi_model``. Now uses the fast kind.

Changed
~~~~~~~

-  ``modelling`` module got an overhaul. It now consists of smaller,
   self-consistent classes that are based on ``attrs`` and are read-only
   (but clonable). There is an explicit split between a ``Model`` and a
   ``FixedLinearModel`` which is defined at some coordinates ``x``,
   allowing the latter to provide speedups for fitting lots of data at
   the same coordinates. They are also YAML-read/writable.

v2.1.0
------

Added
~~~~~

-  ``FourierDay`` linear model

Fixed
~~~~~

-  models were being mutated in ``ModelFit``. No more.

v2.0.0
------

Added
~~~~~

-  Ability to pass through LNA S11 options
-  Ability to specify ``t_load`` and ``t_load_ns`` in spectra
-  Easier conversion to ``Calibration`` object from
   ``CalibrationObservation``
-  New ``NoiseWaves`` linear model
-  Modular ``InternalSwitch`` class

Fixed
~~~~~

-  Ensure fitted models are always copied

v1.0.0
------

Added
~~~~~

-  Visualization of ``xrfi_model`` output.
-  Allow ``init_flags`` as input to ``xrfi_model``

Changed
~~~~~~~

-  Performance boost for ``xrfi_model_sweep``
-  Faster modelling in detail
-  Compatibility with ``edges-io`` v1.
-  More consistency in API between different XRFI models.
-  Allow not passing ``spectrum`` to ``xrfi_explicit`` ### Fixed

-  Improvements to typing.
-  Improvements to ``xrfi`` tests.

v0.7.0
------

Added
~~~~~

-  Automatic notebook reports for calibration
-  ``xrfi_model_sweep`` that actually works and is tested.
-  Faster model evaluations

Fixed
~~~~~

-  xRFI doesn't assume that input spectrum is all positive (could be
   residuals, and therefore have negatives).

Version 0.4.0
-------------

Changed
~~~~~~~

-  Much faster modeling via re-use of basis function evaluations

Version 0.3.0
-------------

Changed
~~~~~~~

-  load\_name now always an alias (hot\_load, ambient, short, open)
-  Load.temp\_ave now always the correct one (even for hot load)

Version 0.2.1
-------------

Added
~~~~~

-  Basic tests
-  Travis/tox/codecov setup

Version 0.2.0
-------------

Added
~~~~~

-  Many many many new features, and complete modularisation of code
-  Now based on ``edges-io`` package to do the hard work.
-  Refined most modules to remove redundant code
-  Added better package structure

Version 0.1.0
-------------

-  First working version on github.
