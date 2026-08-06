# Simulatr

Simulatr provides a [gymnasium](https://gymnasium.farama.org/) environment for reinforcement learning using different simulators including:

- [ApsimX](https://docs.apsim.info/)

## Installation

Simulatr can be installed via pip or pixi can be utilized

```
pip install -e .
```

In addition to installing the gym, you will also need to install the models you wish to run (instructions below).

## Model installation

Simulatr will install the required models automatically, but you can also install the models yourself (steps for each model included below). The simulatr CLI includes a utility for performing these steps before runtime

```
python -m simulatr install apsimx
```

## Command line interface

### Running a simulator engine to model completion

```
python -m simulatr run apsimx --crop-name wheat
```

### Running a simulator engine interactively

```
python -m simulatr run apsimx --crop-name wheat --timestep=10  # days
```

## Model installation details

### ApsimX

#### Requirements

- .NET 8.0 SDK library
- Gtk3 and GtkSourceView

A conda environment file is provided to make it easier to install these dependencies.

#### Installation from source

The basic steps for installing ApsimX are:

1. Install the depenedencies above
1. Clone the ApsimX repository from [here](https://github.com/APSIMInitiative/ApsimX).

   ```
   git clone git@github.com:APSIMInitiative/ApsimX.git
   ```

1. Build the ApsimX.sln solution file

    ```
    dotnet build path/to/ApsimX/ApsimX.sln
    ```

1. Point simulatr at the installation

    ```
    python -m simulatr config apsimx --directory path/to/ApsimX
    ```


See instructions [here](https://docs.apsim.info/docs/development/compile) if you encounter errors when installing ApsimX.


## To do list

- [ ] Switch to pydantic
- [ ] Publish to PyPI
- [ ] Publish to conda-forge
- [ ] Move n8n server into this repo or its own?
- [ ] Update n8n tool to use this repo
- [ ] Redeploy