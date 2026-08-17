# To do

## General

- [ ] Fix windows tests on CI
- [ ] Add soil file parsing
- [ ] Improve human readable prompt for CLI and fix set/get description
- [ ] Fix the parameter order for the apsimx form so that crop/variety first

## Server

### Short term

- [ ] Mark file parameters with os.PathLike
- [ ] Handle file uploads in n8n form generation (weather, soil, model files)
- [ ] Add bearer token authentication when using pod deployment & redeploy since the pod deployment does not seem to implement beam bearer credential
- [ ] Update CLI docs
- [ ] Fix n8n update call

### Long term

- [ ] Improved documentation
- [ ] Endpoint for getting a list of state variables (this would require parsing C# files)
- [ ] Auto test API examples/defaults
- [ ] Add option to upload a crop model file?
