# To do

## General

- [ ] Add soil file parsing
- [ ] Improve human readable prompt for CLI and fix set/get description
- [ ] Fix bugs in piecemeal construction of model
- [ ] Update default for_example to False and add it to the n8n form

## Server

### Short term

- [ ] Mark file parameters with os.PathLike
- [ ] Handle file uploads in n8n form generation (weather, soil, model files)
- [ ] Add bearer token authentication when using pod deployment & redeploy since the pod deployment does not seem to implement beam bearer credential

### Long term

- [ ] Improved documentation
- [ ] Endpoint for getting a list of state variables (this would require parsing C# files)
- [ ] Auto test API examples/defaults
