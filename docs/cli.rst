======================
Command Line Interface
======================

``simulatr`` ships with a command line interface for configuring the
package, installing simulators, creating model input files, and running
simulations. The commands below all use the :mod:`simulatr.cli` module.

.. contents::
   :local:
   :depth: 2

Invoking the CLI
================

If ``simulatr`` is installed into your environment, the ``simulatr``
console script is available on your ``PATH``:

.. code-block:: console

   $ simulatr --help

The same interface can be invoked as a module, which works without a
console script:

.. code-block:: console

   $ python -m simulatr --help

Both forms accept the same subcommands::

   usage: simulatr [-h] {config,install,create,run} ...

Overview
========

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Command
     - Description
   * - ``simulatr config``
     - Set a configuration option in a configuration file.
   * - ``simulatr install``
     - Install or register a simulator (e.g. ApsimX).
   * - ``simulatr create``
     - Create a simulator model input file.
   * - ``simulatr run``
     - Run a simulation.

Configuration
=============

``simulatr config <section> <name> <value>`` writes a single option to a
configuration file. Options for the ``directories`` and ``files``
sections are converted to absolute paths automatically.

Configuration files live in ``.simulatr.ini`` files that are read from
(and written to) several locations:

.. code-block:: console

   $ simulatr config directories apsimx /Users/me/ApsimX
   Set apsimx in the "directories" section of
   "/Users/me/my-project/.simulatr.ini" to "/Users/me/ApsimX"

By default the option is written to the most local configuration file
(the current working directory). The ``--level`` flag selects a
different location:

.. code-block:: console

   $ simulatr config directories apsimx /Users/me/ApsimX --level local
   $ simulatr config directories apsimx /Users/me/ApsimX --level user
   $ simulatr config directories apsimx /Users/me/ApsimX --level env

The ``directories`` section supports the following options:

* ``output`` -- Directory for simulation output.
* ``models`` -- Base directory where simulator models are installed.
* ``apsimx`` -- Directory containing the ApsimX installation.
* ``nasa_power_weather_data`` -- Cache directory for downloaded NASA
  POWER weather data.

.. _cli-install:

Installing a simulator
======================

``simulatr install apsimx`` installs ApsimX into the directory given by
the ``apsimx`` configuration option (by default
``./models/apsimx``), prompting for confirmation:

.. code-block:: console

   $ simulatr install apsimx
   Install the apsimx model into ".../models/apsimx"? [Y/n]

Use ``--directory`` to install into a different location and record it
in the configuration at the same time:

.. code-block:: console

   $ simulatr install apsimx --directory /Users/me/ApsimX

If a valid installation already exists at the configured location, the
command simply registers it and returns.

Creating model input files
==========================

``simulatr create apsimx <crop>`` generates an ApsimX ``.apsimx`` model
input file for the requested crop. The crop name is case-insensitive.

.. code-block:: console

   $ simulatr create apsimx wheat
   Created input file "Wheat-Generated-Interactive.apsimx"

The generated file supports interactive interventions such as sowing,
harvesting, irrigation, and fertilization.

Choosing the output path
------------------------

.. code-block:: console

   $ simulatr create apsimx wheat --dst ./wheat.apsimx
   Created input file "./wheat.apsimx"

.. note::
   Pass a path that includes a directory component (for example
   ``./wheat.apsimx``). A bare filename such as ``--dst wheat.apsimx``
   is resolved relative to the ``Examples`` subdirectory and will fail
   unless that directory exists.

Selecting the available actions
-------------------------------

The ``--actions`` option restricts which interventions are included in
the interactive file:

.. code-block:: console

   $ simulatr create apsimx wheat \
       --actions sow harvest nitrogen \
       --dst ./wheat.apsimx

Copying an example
------------------

.. code-block:: console

   # Copy the bundled Wheat example
   $ simulatr create apsimx wheat --from-example

   # Copy a specific example file
   $ simulatr create apsimx wheat --from-example /path/to/example.apsimx

When no example is requested the file is generated from a crop template
instead.

Overwriting existing files
--------------------------

.. code-block:: console

   $ simulatr create apsimx wheat --dst ./wheat.apsimx
   $ simulatr create apsimx wheat --dst ./wheat.apsimx
   RuntimeError: Model file already exists: "./wheat.apsimx"
   $ simulatr create apsimx wheat --dst ./wheat.apsimx --overwrite
   Created input file "./wheat.apsimx"

.. note::
   The ``--interactive`` flag is accepted for compatibility but is
   currently a no-op: files generated by ``create`` are always
   interactive.

Running a simulation
====================

``simulatr run apsimx`` starts the ApsimX ZMQ server and runs a
simulation using the ApsimX gymnasium environment. A working ApsimX
installation is required (see :ref:`the install command <cli-install>`
and the README).

Run a simulation from a crop name:

.. code-block:: console

   $ simulatr run apsimx --crop-name wheat

Or from an existing model input file:

.. code-block:: console

   $ simulatr run apsimx --model-file ./wheat.apsimx

Running interactively
---------------------

By default the simulation runs continuously to completion
(``--timestep 0``). A positive ``--timestep`` pauses the simulation
every ``timestep`` days and asks the user what action to take next. At
each pause a prompt is generated from the current observation and the
available actions:

.. code-block:: console

   $ simulatr run apsimx --crop-name wheat --timestep 7

The prompt describes the current state of the simulation and lists the
available actions, along with the exact response format to use. Type
your chosen action and press Enter. The response is parsed and applied
to the simulation before it advances to the next step, for example::

   <answer>Apply 2 kg/ha of nitrogen fertilizer in the form of NO3</answer>

Observations
------------

Use ``--state-variables`` to select which simulation state variables are
included in the observation at each step:

.. code-block:: console

   $ simulatr run apsimx --crop-name wheat \
       --state-variables \
           "[Clock].Today" \
           "[CROP].Phenology.CurrentStageName" \
           "[CROP].Total.Wt" \
           "[Weather].Rain" \
           "[Soil].Water.PAW"

Allowing actions
----------------

The ``--actions`` option limits which interventions the user can choose
from at each step:

.. code-block:: console

   $ simulatr run apsimx --crop-name wheat \
       --actions sow harvest nitrogen

Logging
-------

The ``--log-level`` option controls the verbosity of logging and
``--log-file`` redirects log messages to a file. With no argument,
``--log-file`` writes to ``<simulator>.log`` in the current directory:

.. code-block:: console

   $ simulatr run apsimx --crop-name wheat --log-file
   Log being written to "apsimx_wheat.log"

   $ simulatr run apsimx --crop-name wheat \
       --log-file ./run.log --log-level DEBUG
