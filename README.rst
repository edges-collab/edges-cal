edges\_cal
==========

|Build Status| |codecov| |Code style: black| |RTD|

**Calculate calibration coefficients for EDGES Calibration
Observations**

Installation
------------

Download/clone the repo and do

::

    pip install .

in the top-level directory (optionally add an ``-e`` for develop-mode).
Preferably, do this in an isolated python/conda environment.

Usage
-----

CLI
~~~

There is a very basic CLI set up for running a full calibration pipeline
over a set of data. To use it, do

::

    $ edges-cal run --help

Multiple options exist, but the only ones required are ``CONFIG`` and
``PATH``. The first should point to a YAML configuration for the run,
and the second should point to a directory in which exists ``S11``,
``Resistance`` and ``Spectra`` folders. Thus:

::

    $ edges-cal run ~/config.yaml .

will work if you are in such a directory.

The ``config.yaml`` consists of a set of parameters passed to
``edges_cal.CalibrationObservation``. See its docstring for more
details.

In addition, you can run a "term sweep" over a given calibration,
iterating over number of Cterms and Wterms until some threshold is met.
This uses the same configuration as ``edges-cal run``, but you can pass
a maximum number of C and W-terms, along with a threshold at which to
stop the sweep (this is a threshold in absolute RMS over degrees of
freedom). This will write out a ``Calibration`` file for the "best" set
of parameters.

You can also create full Jupyter notebook reports (and convert them to
PDF!) using the CLI. To get this to work, you must install ``edges-cal``
with ``pip install edges-cal[report]``. Then you must do the following:

1. Activate the environment you wish to use to generate the reports
   (usually ``conda activate edges``)
2. Run
   ``python -m ipykernel install --user --name edges --display-name "edges"``

Note that in the second command, calling it "edges" is necessary
(regardless of the name of your environment!).

Now you can run

::

    $ edges-cal report PATH --config ~/config.yaml

(obviously there are other parameters -- use ``edges-cal report --help``
for help). The ``PATH`` should again be a calibration observation
directory. The config can be the same file as in ``edges-cal run``, and
is optional. By default, both a notebook and a PDF will be produced, in
the ``outputs/`` directory of the observation. You can turn off the PDF
production with a ``-R`` flag.

Similarly, you can *compare* two observations as a report notebook with

::

    $ edges-cal compare PATH COMPARE --config ~/config.yaml --config-cmp ~/config.yaml

This is intended to more easily show up what might be wrong in an
observation, when compared to a "golden" observation, for example.

Using the Library
~~~~~~~~~~~~~~~~~

To import:

::

    import edges_cal as ec

Most of the functionality is highly object-oriented, and objects exist
for each kind of data/measurement. However, there is a container object
for all of these, which manages them. Thus you will typically use

::

    >>> calobs = ec.CalibrationObservation(path="path/to/top/level")

Several other options exist, and they have documentation that you can
access interactively by using

::

    >>> help(ec.CalibrationObservation)

The most relevant attributes are the (lazily-evaluated) calibration
coefficient models:

::

    >>> plt.plot(calobs.freq.freq, calobs.C1())

the various plotting routines, eg.

::

    >>> calobs.plot_coefficients()

and the calibrate/decalibrate methods:

::

    >>> calibrated_temp = calobs.calibrate("ambient")

Note that this final method can be applied to any ``LoadSpectrum`` --
i.e. you can pass in field observations, or an antenna simulator.

.. |Build Status| image:: https://travis-ci.org/edges-collab/edges-cal.svg?branch=master
   :target: https://travis-ci.org/edges-collab/edges-cal
.. |codecov| image:: https://codecov.io/gh/edges-collab/edges-cal/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/edges-collab/edges-cal
.. |Code style: black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black
.. |RTD| image:: https://readthedocs.org/projects/edges-cal/badge/?version=latest
   :target: https://edges-cal.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status
