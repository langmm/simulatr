# Simulatr

[![PyPI version](https://img.shields.io/pypi/v/simulatr.svg)](https://pypi.org/project/simulatr/)
[![Python versions](https://img.shields.io/pypi/pyversions/simulatr.svg)](https://pypi.org/project/simulatr/)
[![PyPI downloads](https://img.shields.io/pypi/dm/simulatr.svg)](https://pypi.org/project/simulatr/)
[![License](https://img.shields.io/pypi/l/simulatr.svg)](https://github.com/langmm/simulatr/blob/main/LICENSE.txt)
[![Tests](https://github.com/langmm/simulatr/actions/workflows/runtests.yml/badge.svg)](https://github.com/langmm/simulatr/actions/workflows/runtests.yml)
[![Conda build](https://github.com/langmm/simulatr/actions/workflows/build-conda.yml/badge.svg)](https://github.com/langmm/simulatr/actions/workflows/build-conda.yml)
[![Docs](https://github.com/langmm/simulatr/actions/workflows/publish-docs.yml/badge.svg)](https://langmm.github.io/simulatr/)

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
python -m simulatr install
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
