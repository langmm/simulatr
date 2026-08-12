=================
Simulator Servers
=================

REST API
--------

To launch the server run the following from the n8n_tool directory::

  fastapi run --host 0.0.0.0 --port 5000 main.py apsimx

TODO: Correct this w/ server entry point for package and add CLI argument for specifying the model to host

or via pixi...::

  pixi run -e py313 run-server --simulator apsimx

To run the tests::

  pixi run -e py313 test -svx tests/test_server.py


Docker
------

To build the docker image this run the following from the simulatr root directory::

  docker build -f Dockerfile.n8n -t apsimx .


TODO: Build arg to pass model


or via pixi...::

  pixi run -e py313 build-docker --simulator apsimx


To run the server in the docker image::

  docker run -p 5000:8000 -d apsimx

or via pixi...::

  pixi run -e py313 run-server-docker --simulator apsimx

  
To select a different host/container port::

  docker build -f n8n_tool/Dockerfile --build-arg APP_PORT=${CONTAINER_PORT} -t apsimx .
  docker run -p ${HOST_PORT}:${CONTAINER_PORT} -d apsimx


Beam Deployment
---------------

To deploy pod based application to beam.cloud::

  beam deploy n8n_tool/beam_pod.py:pod


To serve docker/python based ASGI application.::

  beam serve n8n_tool/beam_asgi.py:web_server
  beam serve n8n_tool/beam_docker.py:web_server

The asgi version does not work currently and causes the following error::

  /micromamba/envs/beta9/bin/python3: Error while finding module specification for 'beta9.runner.serve' (ModuleNotFoundError: No module named 'beta9')

To run the tests using the deployment on beam::

  export SIMULATR_REMOTE_SERVER_ADDRESS=<BEAM_URL>
  pytest -svx tests/test_server.py


n8n Tool
--------

The ``simulatr.n8n`` module provides CLI tools for managing an n8n tool that uses the beam service.

To access the n8n api, a valid API key must be passed via the ``X_N8N_API_KEY`` environment variable. The address of the service on beam can be provided via the ``SIMULATR_REMOTE_SERVER_ADDRESS`` environment variable or via the ``--publish-for-address`` CLI argument::

  export X_N8N_API_KEY=<N8N_CREDENTIALS>
  export SIMULATR_REMOTE_SERVER_ADDRESS=<BEAM_URL>


To create an n8n tool that uses the apsimx simulator service::

  python -m simulatr n8n apsimx create --name start


To update an existing n8n tool that uses the apsimx model service::

  python -m simulatr n8n apsimx update --name start


