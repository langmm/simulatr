# To do

## General

- [ ] Fix windows tests on CI
- [ ] Add soil file parsing
- [ ] Improve human readable prompt for CLI and fix set/get description
- [ ] Add engine methods that can serve as decorators to add endpoints to a fastapi application
- [ ] Move n8n server into this repo or its own?
- [ ] Update n8n tool to use this repo
- [ ] Redeploy to beam
- [ ] Update n8n server

## Server

### Short term

- [ ] Finish adding server interface to simulatr repo w/ tests
- [ ] Handle file uploads in n8n form generation (weather, soil, model files)
- [ ] Add bearer token authentication when using pod deployment & redeploy since the pod deployment does not seem to implement beam bearer credential
- [ ] Add support for providing a weather/soil file via upload

### Long term

- [ ] Improved documentation
- [ ] Test build with different version of Python?
- [ ] Endpoint for getting a list of state variables (this would require parsing C# files)
- [ ] Auto test API examples/defaults
- [ ] Add option to upload a crop model file?
