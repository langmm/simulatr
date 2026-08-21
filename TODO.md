# To do

## General

- [ ] Fix readme inclusion in PyPI description
- [ ] Fix soil file parsing of XML data from SSURGO
- [ ] Improve human readable prompt for CLI and fix set/get description
- [ ] Add lookup for crop costs from USDA NASS public API and plot that for intercrop comparison of yield
- [ ] Update crop panel to go over two years to allow for variation in crop sowing times base on weather
- [ ] Verify that AutoSow makes sense for all crops (or is parametrized correctly)

## Server

### Short term

- [ ] Mark file parameters with os.PathLike
- [ ] Handle file uploads in n8n form generation (weather, soil, model files)
- [ ] Add bearer token authentication when using pod deployment & redeploy since the pod deployment does not seem to implement beam bearer credential

### Long term

- [ ] Improved documentation
- [ ] Endpoint for getting a list of state variables (this would require parsing C# files)
- [ ] Auto test API examples/defaults
