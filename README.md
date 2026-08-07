Simulatr provides a [gymnasium](https://gymnasium.farama.org/) environment for reinforcement learning using different simulators including:

- [ApsimX](https://docs.apsim.info/)

# Installation

Simulatr can be installed via pip or pixi can be utilized

```
pip install -e .
```

In addition to installing the gym, you will also need to install the simulators you wish to run.

# Simulator installation

Simulatr will install the required simulators automatically, but you can also install the simulators yourself via the simulatr CLI before running a simulator

```
python -m simulatr install apsimx
```

# Running a simulator

## Running to completion

```
python -m simulatr run apsimx --crop-name wheat --from-example
```

## Running interactively

```
python -m simulatr run apsimx --crop-name wheat --from-example --timestep=10  # days
```

Additional documentation can be found [here](https://langmm.github.io/simulatr/),
