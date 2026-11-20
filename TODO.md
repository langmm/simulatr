# To do

## In progress
- Fill in missing or incomplete docstrings in the BaseSchedule, ModelOperation, and ModelSchedule classes in simulatr/base.py
- Update pixi tasks to split dev dependencies between test, doc, and conda (lint?)

## General

- [ ] Generalize BaseDataFile
  - Answer test resampling of soil depths
  - Generation of soil profile
  - Setter for derived properties
  - Class defined value indicating missing data (so it can be replaced with NAN)
- [ ] Fix readme inclusion in PyPI description
- [ ] Improve human readable prompt for CLI and fix set/get description
- [ ] Add lookup for crop costs from USDA NASS public API and plot that for intercrop comparison of yield
- [ ] Update crop panel to go over two years to allow for variation in crop sowing times base on weather
- [ ] Verify that AutoSow makes sense for all crops (or is parametrized correctly)
- [ ] Non-interactive action schedule for ApsimX
- [ ] Add DSSAT

## ApsimX

### Soil

- [ ] Get initial water content from NASA POWER or SMAP L4

### WeedCycle in simulatr/apsimx_data/WeedCycle.json

- [ ] Dormancy rate based on light (other resource competition?)
- [ ] Check that residue is treated correctly on weed "harvest"
- [ ] Multiple cohorts w/ interval or ...?

## Server

### Short term

- [ ] Mark file parameters with os.PathLike
- [ ] Handle file uploads in n8n form generation (weather, soil, model files)
- [ ] Add bearer token authentication when using pod deployment & redeploy since the pod deployment does not seem to implement beam bearer credential

### Long term

- [ ] Improved documentation
- [ ] Endpoint for getting a list of state variables (this would require parsing C# files)
- [ ] Auto test API examples/defaults
