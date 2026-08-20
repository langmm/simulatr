# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

<!-- insertion marker -->
## [v0.1.0](https://github.com/langmm/simulatr/releases/tag/v0.1.0) - 2026-08-20

<small>[Compare with v0.0.1](https://github.com/langmm/simulatr/compare/v0.0.1...v0.1.0)</small>

### Features

- Tools for hosting simulator engines as fastapi applications on beam
- Tools for creating n8n tools for simulator engine applications
- CLI for managine simulator applications and n8n tools
- Support for soil file input to crop models
- Utilities for iterating over pydantic fields that checks for annotations that should be skipped (used for n8n forms and CLI)

### Added

- Added flag to prevent hang on apsimx termination due to failed connection ([27da4bf](https://github.com/langmm/simulatr/commit/27da4bfea8163c0aa478c7940ca5dd9c4c236a8b) by Meagan Lang).
- Added option to force reinstallation of simulator ([0f6846c](https://github.com/langmm/simulatr/commit/0f6846cfed36623406c88e51b119e5907b5b3420) by Meagan Lang).
- Added dockerfile and pixi tasks for building/running docker image containing server ([4128bbf](https://github.com/langmm/simulatr/commit/4128bbf9370a8da22075687789d31410e808a60f) by Meagan Lang).
- Added output_vars & timestep fields to model engine for use with server and update how the resume methods works to allow for a variable timestep ([0cb714e](https://github.com/langmm/simulatr/commit/0cb714eba254c59b06e96d70dc499bacee571666) by Meagan Lang).
- Added soil file base class and ISRIC class for getting data from ISRIC SoilGrids (the REST API is not stable atm so this should not be the default soil data source) Begin matching data model to server Move data classes into data module and add a BaseDataFile class for more uniform treatment of conversion between data files and caching ([bbac3d7](https://github.com/langmm/simulatr/commit/bbac3d7817aae11f09a3d6310b9df1f6d7963c8c) by Meagan Lang).
- Added apsimx conda dependencies to recipe ([caa6fa2](https://github.com/langmm/simulatr/commit/caa6fa23c7b50596556b732206b0ff288f3043bf) by Meagan Lang).
- Added summary job to conda build GHA workflow for use in rulesets Set python_min in recipe context to allow building outside conda-forge Lint ([a5509c1](https://github.com/langmm/simulatr/commit/a5509c149440a2dc9fa53c0aaf685ab60a58ff01) by Meagan Lang).
- Added docs on release process and added badges to README.md ([0ce6982](https://github.com/langmm/simulatr/commit/0ce69826c7705a7b532a2bc012f800df822e100c) by Meagan Lang).
- Added CHANGELOG ([79a894d](https://github.com/langmm/simulatr/commit/79a894d9eb5e6726e0b4292afa0de02b58385541) by Meagan Lang).

## [v0.0.1](https://github.com/langmm/simulatr/releases/tag/v0.0.1) - 2026-08-07

<small>[Compare with first commit](https://github.com/langmm/simulatr/compare/324cfff51d20d4a57f146604e5bbe60959c564f5...v0.0.1)</small>

Initial release migrating from [ApsimXGym](https://github.com/langmm/ApsimX/tree/yggdrasil/ApsimXGym).

### Features included

- Support for ApsimX simulator engine including installation from GitHub
- Classes for running simulator engines interactively
- Classes for running simulator engines as [gymnasium](https://gymnasium.farama.org/) RL environments
- Ability to generated LLM prompts for simulator based RL environments for [AgriManager](https://github.com/CHIGUI0/AgriManager)
- CLI for running simulators and managing simulatr configuration
- pixi environments and tasks
- Sphinx based [documentation](https://langmm.github.io/simulatr) with GHA workflow to automatically updated
- Conda recipe with GHA workflow to build recipe
- Tests for all classes with GHA workflow to run tests
