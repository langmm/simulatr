============
Usage Guide
============

The :mod:`simulatr.apsimx` module provides everything needed to create,
modify, and run `ApsimX`_ crop simulation models and to use them as
gymnasium reinforcement-learning environments.

.. _ApsimX: https://apsiminitiative.github.io/apsimx

.. contents::
   :local:
   :depth: 2

.. _apsimx-files:

Model input files
=================

An :class:`~simulatr.apsimx.ApsimXFile` is a container for
manipulating ``.apsimx`` model input files. Files can be generated from
a crop name, copied from a bundled example, or read from an existing
path.

Generating a new file from a crop name
--------------------------------------

.. code-block:: python

   from simulatr.apsimx import ApsimXFile

   # List the crops that can be simulated
   print(ApsimXFile.available_crops())

   # List the cultivars for a given crop
   print(ApsimXFile.available_cultivars("Wheat"))

   # Generate a new Wheat model input file
   model = ApsimXFile.from_crop_name("Wheat")
   print(model.fname)      # "Wheat-Generated.apsimx"
   print(model.crop_name)  # "wheat"

   # Write the file to disk (the model file has already been created)
   model.write(overwrite=True)

Generating a file with interactive actions
------------------------------------------

Passing ``interactive=True`` (or providing ``actions``) generates a
model file that supports runtime interventions such as sowing,
harvesting, irrigation, and fertilization.

.. code-block:: python

   model = ApsimXFile.from_crop_name(
       "Wheat",
       interactive=True,
       actions=["sow", "harvest", "nitrogen", "irrigate"],
   )
   print(model.fname)        # "Wheat-Generated-Interactive.apsimx"
   print(model.is_interactive)

Copying a bundled example
-------------------------

.. code-block:: python

   # Locate the example model file in the ApsimX installation
   src = ApsimXFile.find_example("Wheat")

   # Copy the example and make it interactive
   model = ApsimXFile.from_example(
       src,
       interactive=True,
       actions=["sow", "harvest"],
   )
   print(model.fname)
   print(model.is_interactive)

Reading an existing file
------------------------

.. code-block:: python

   model = ApsimXFile("path/to/my-model.apsimx")
   print(model.exists)
   print(model.crop_name)     # crop name inferred from the file
   print(model.crop_variety)  # e.g. "Hartog"
   print(model.location)      # e.g. "-27.581836°N, 151.320206°E"
   print(model.field_area)    # area in hectares

Getting and setting parameters
------------------------------

Parameters are addressed by name and can be read with :meth:`get` and
written with :meth:`set`.

.. code-block:: python

   print(model.get("latitude"))   # e.g. -27.581836
   print(model.get("longitude"))

   model.set("latitude", -27.5)
   model.write(overwrite=True)

   # Values that do not exist are an error unless a default is given
   print(model.get("nonexistent", default="not set"))

Moving files
------------

.. code-block:: python

   # Rename/move the file (returns the new path)
   new_fname = model.move(dst="wheat-copy.apsimx")
   print(new_fname)

   # Or place it in a different directory with a suffix
   model.move(directory="outputs", suffix="v2")

.. note::
   ``crop_name`` and ``crop_variety`` describe the *target* crop. Files
   generated with :meth:`~simulatr.apsimx.ApsimXFile.from_crop_name`
   produce a bare simulation skeleton, so parameters such as ``latitude``
   are not present until you set them or start from an example file.

File nodes
==========

An ``.apsimx`` file is a tree of :class:`~simulatr.apsimx.ApsimXFileNode`
objects. You generally do not need to work with nodes directly, but they
are available for low-level manipulation.

.. code-block:: python

   from simulatr.apsimx import ApsimXFileNode

   # Build a node from a data resource bundled with ApsimX
   clock = ApsimXFileNode.from_data("Clock")
   print(clock.absolute_path)   # e.g. "[Clock]"

   # Build a node from a model type
   soil = ApsimXFileNode.from_param(
       "Models.Soils.Arbitrator.SoilArbitrator, Models")
   print(soil["Name"])

   # Load a node directly from a JSON file
   node = ApsimXFileNode.from_file("some-node.json")

   # Navigate the tree
   for child in soil.children:
       print(child.absolute_path)

Weather files
=============

:class:`~simulatr.apsimx.ApsimXWeatherFile` reads and writes ApsimX
weather files (``.met``) and can download NASA POWER data for any
location.

.. code-block:: python

   import datetime

   from simulatr.apsimx import ApsimXWeatherFile

   weather = ApsimXWeatherFile.from_location(
       latitude=-27.58,
       longitude=151.32,
       start_date=datetime.date(1981, 1, 1),
       end_date=datetime.date(1981, 12, 31),
   )
   print(weather.fname)
   print(weather.dates[0])   # np.datetime64('1981-01-01')
   print(weather.latitude, weather.longitude)

.. note::
   ``from_location`` caches downloaded data on disk and reuses it if a
   cached file already covers the requested dates.

Soil files
==========

:class:`~simulatr.apsimx.ApsimXSoilFile` reads and writes ApsimX soil
files (``.soil.json``, e.g. ``simulatr/data/Soil.json``) and can
generate a synthetic soil profile for any location from ISRIC SoilGrids
data.

.. code-block:: python

   from simulatr.apsimx import ApsimXSoilFile

   soil = ApsimXSoilFile.from_location(
       latitude=-27.58,
       longitude=151.32,
   )
   print(soil.fname)
   print(soil.depths)   # [(0, 5), (5, 15), (15, 30), (30, 60), ...]
   print(soil.latitude, soil.longitude)

   # Load the example soil file bundled with simulatr
   example = ApsimXSoilFile("simulatr/data/Soil.json")
   print(example.depths)

.. note::
   ``from_location`` downloads ISRIC SoilGrids data, converts it to the
   ApsimX soil format using the Saxton and Rawls (2006) pedotransfer
   functions, and caches the result on disk so it is only generated
   once per location.

Running a simulation
====================

An :class:`~simulatr.apsimx.ApsimXEngine` manages communication with an
ApsimX ZMQ server running in a separate process. A working ApsimX
installation is required (see the README for install instructions).

.. code-block:: python

   from simulatr.apsimx import ApsimXEngine

   # Check the installation
   print(ApsimXEngine.is_installed())
   print(ApsimXEngine.model_dir())

   # The engine can be configured from a crop name...
   engine = ApsimXEngine(
       crop_name="Wheat",
       crop_variety="Hartog",
       start_time="1981-06-01",
       end_time="1981-11-30",
       latitude=-27.58,
       longitude=151.32,
   )

   # ...or from an existing model file
   engine = ApsimXEngine(model_file="wheat.apsimx")

Starting and running
--------------------

.. code-block:: python

   import datetime

   engine.start()

   # Run to a specific date
   engine.fast_forward(datetime.datetime(1981, 6, 1))

   # Sow the crop at the current date
   engine.act("sow", crop_name="Wheat", crop_variety="Hartog")

   # Run forward by a timedelta
   engine.fast_forward(datetime.timedelta(days=30))

   # Read state variables
   vars = engine.getvars([
       "[Clock].Today",
       "[CROP].Phenology.CurrentStageName",
       "[CROP].Total.Wt",
       "[Weather].Rain",
       "[Soil].Water.PAW",
   ])
   print(vars)

   # Apply an intervention
   engine.act("nitrogen", amount=2.0)   # kg/ha of N fertilizer
   engine.act("irrigate", amount=5.0)   # mm of water

   # Run the rest of the season and harvest
   engine.fast_forward()
   engine.act("harvest", crop_name="Wheat")

   # End the simulation and shut down the server
   engine.act("terminate")
   engine.stop(cleanup=True)

.. note::
   Actions are applied at the *current* simulation date, so call
   :meth:`~simulatr.apsimx.ApsimXEngine.fast_forward` before acting to
   schedule interventions. Available actions and their parameters are
   listed in :attr:`~simulatr.apsimx.ApsimXEngine.AVAILABLE_ACTION_MAP`.

Other engine methods
--------------------

.. code-block:: python

   engine.is_running          # True while the simulation is running
   engine.simulation_date     # current simulation date/time
   engine.resume(wait=True)   # resume a paused simulation
   engine.stop()              # stop the server without cleaning up files

Reinforcement learning environment
==================================

:class:`~simulatr.apsimx.ApsimXEnv` wraps an
:class:`~simulatr.apsimx.ApsimXEngine` as a gymnasium
environment. The observation is the set of ``output_vars`` (default:
crop yield plus a selection of crop, soil, and weather state
variables), and the action space is built from the configured
``actions`` (default: ``["nitrogen", "irrigate"]``).

Constructing an environment
---------------------------

.. code-block:: python

   from simulatr.apsimx import ApsimXEnv

   env = ApsimXEnv(
       crop_name="Wheat",
       crop_variety="Hartog",
       start_time="1981-06-01",
       end_time="1981-11-30",
       latitude=-27.58,
       longitude=151.32,
   )

   print(env.observation_space)  # gymnasium Box/Dict
   print(env.action_space)       # gymnasium Box/Dict
   print(env.action_map)         # mapping of action ids to actions

   obs, info = env.reset(seed=42)
   print(obs)

   action = env.action_space.sample()
   obs, reward, terminated, truncated, info = env.step(action)

   env.close()

Interacting manually
--------------------

For human-in-the-loop or agent-verification workflows, the environment
can be made interactive and stepped from the terminal:

.. code-block:: python

   env = ApsimXEnv(
       crop_name="Wheat",
       interactive=True,
       actions=["sow", "harvest", "nitrogen", "irrigate"],
   )
   env.create_interactive_for_human()
   obs, info = env.reset()
   obs, reward, terminated, truncated, info = env.step(action)

Options
-------

.. code-block:: python

   # Larger intervention interval (days between automatic actions)
   env = ApsimXEnv(crop_name="Wheat", intervention_interval=14)

   # Custom reward variable
   env = ApsimXEnv(
       crop_name="Wheat",
       revenue_var={"name": "[CROP].Grain.Total.Wt", "cost": 0.5},
   )

   # Control the discretization of continuous actions
   env = ApsimXEnv(crop_name="Wheat", num_levels=0, exclusive=False)

LLM prompt generation
=====================

:class:`~simulatr.apsimx.ApsimXLLMPromptGenerator` turns an
environment's observations and action space into text prompts that can
be used with a large language model, and parses the model's responses
back into actions.

.. code-block:: python

   from simulatr.apsimx import ApsimXLLMPromptGenerator

   generator = ApsimXLLMPromptGenerator.from_env(env)

   # System prompt describing the environment and action space
   system_prompt = generator.get_system_prompt()
   print(system_prompt)

   # Turn prompt built from a single observation
   obs, info = env.reset(seed=42)
   turn_prompt = generator.get_turn_prompt(obs)
   print(turn_prompt)

   # Have an LLM respond, then parse its response back into an action
   llm_response = "Apply 2.0 kg/ha of nitrogen."
   action = generator.parse_action_response(llm_response)
   print(action)

   # Describe an action id or parameter set in natural language
   print(generator.describe_action(action))
   print(generator.describe_action(env.action_map.id("nitrogen")))
