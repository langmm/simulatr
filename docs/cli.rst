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

   usage: simulatr [-h] {config,install,create,run,serve,n8n} ...

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
   * - ``simulatr serve``
     - Launch one or more simulators as a FastAPI application.
   * - ``simulatr n8n``
     - Manage n8n tools that expose simulator endpoints.

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
* ``isric_soil_data`` -- Cache directory for downloaded ISRIC soil
  data.
* ``scratch`` -- Temporary directory for intermediate files generated
  by the n8n tool utilities.

.. _cli-install:

Installing a simulator
======================

``simulatr install`` installs one or more simulators. Without arguments
every registered simulator is installed:

.. code-block:: console

   $ simulatr install

Select a specific simulator with ``--simulator``:

.. code-block:: console

   $ simulatr install --simulator apsimx

Use ``--directory`` to install into a custom location and record it
in the configuration at the same time. ``--directory`` cannot be used
when more than one simulator is specified:

.. code-block:: console

   $ simulatr install --simulator apsimx --directory /Users/me/ApsimX

Skip the confirmation prompt with ``--always-yes`` and force
reinstallation of an already-installed simulator with ``--force``:

.. code-block:: console

   $ simulatr install --simulator apsimx --always-yes --force

Arguments
---------

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Flag
     - Description
   * - ``--simulator SIM [SIM ...]``
     - Simulator(s) to install. Defaults to all registered simulators.
   * - ``--directory DIR``
     - Custom install directory. Cannot be used with more than one
       simulator.
   * - ``--always-yes``
     - Skip the confirmation prompt.
   * - ``--force``
     - Force reinstallation even if the simulator is already
       installed.

If a valid installation already exists at the configured location the
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

.. _cli-serve:

Serving simulators
==================

``simulatr serve`` launches one or more simulators as a FastAPI web
application. By default every installed simulator is exposed:

.. code-block:: console

   $ simulatr serve --simulator apsimx

Select simulators explicitly with ``--simulator``, bind to a specific
host and port, and optionally allow remote shutdown:

.. code-block:: console

   $ simulatr serve --simulator apsimx \
       --host 0.0.0.0 --port 8000 \
       --allow-shutdown

Arguments
---------

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Flag
     - Description
   * - ``--simulator SIM [SIM ...]``
     - Simulator(s) to expose via REST endpoints. Defaults to all
       installed simulators.
   * - ``--port PORT``
     - Port that the application is served on (default: ``5000``).
   * - ``--host HOST``
     - Host address to bind to (default: ``0.0.0.0``).
   * - ``--log-file FILE``
     - File to write log messages to.
   * - ``--log-level LEVEL``
     - Logging verbosity. Choices: ``NOTSET``, ``DEBUG``, ``INFO``,
       ``WARNING``, ``ERROR``, ``CRITICAL`` (default: ``INFO``).
   * - ``--allow-shutdown``
     - Include a ``/shutdown`` endpoint that allows the client to
       stop the server.

.. _cli-n8n:

n8n tools
=========

``simulatr n8n`` manages n8n workflow tools that wrap simulator REST
APIs as web forms. A valid n8n API key must be available in the
``X_N8N_API_KEY`` environment variable.

.. code-block:: console

   $ export X_N8N_API_KEY=<key>
   $ simulatr n8n apsimx create --name start \
       --publish-for-address <service-address>

The ``SIMULATR_REMOTE_SERVER_ADDRESS`` environment variable is used as
a fallback for the service address when ``--publish-for-address`` is
not provided.

Common arguments
-----------------

These arguments are available for every ``n8n`` subcommand
(``create``, ``update``, ``remove``, ``query``):

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Flag
     - Description
   * - ``simulator``
     - Name of the simulator to manage tools for.
   * - ``--name NAME [NAME ...]``
     - Entry point(s) to act on. Choices: ``start``,
       ``start-interactive``. Defaults to both entry points if not
       specified.
   * - ``--toolname NAME``
     - Explicit name of the n8n tool. When provided for ``query`` or
       ``remove``, ``--name`` is not required.
   * - ``--output-tool FILE``, ``--output FILE``
     - Output the tool summary to a JSON file. With no value, a
       default filename is used.
   * - ``--verbose``
     - Print every REST API request and response.

create
------

Creates a new n8n tool. Additional arguments:

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Flag
     - Description
   * - ``--publish-for-address ADDR``
     - Base address of the simulator service the tool should call.
   * - ``--overwrite``
     - Remove any existing tool before creating the new one.
   * - ``--update``
     - Update the existing tool instead of failing when a tool with
       the same name already exists.
   * - ``--output-request [FILE]``
     - Output the tool creation request to a JSON file.
   * - ``--output-form [FILE]``
     - Output the form definition to a JSON file.
   * - ``--dry-run``
     - Print requests instead of performing them.

.. code-block:: console

   $ simulatr n8n apsimx create --name start \
       --publish-for-address https://server.example.com \
       --overwrite

update
------

Updates an existing n8n tool. Accepts the same additional arguments
as ``create`` above. The ``--update`` flag is set automatically
so an existing tool is required:

.. code-block:: console

   $ simulatr n8n apsimx update --name start \
       --publish-for-address https://server.example.com

remove
------

Removes an existing n8n tool. Additional arguments:

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Flag
     - Description
   * - ``--dry-run``
     - Print requests instead of performing them.

.. code-block:: console

   $ simulatr n8n apsimx remove --name start

query
-----

Queries the n8n service for tools matching a given name:

.. code-block:: console

   $ simulatr n8n apsimx query --name start
