=================
Simulator Servers
=================

REST API
--------

To launch the server run the following::

  simulatr serve --simulator apsimx

or via pixi...::

  pixi run -e dev python -m simulatr serve --simulator apsimx

To run the tests::

  pixi run -e dev test -svx tests/test_server.py


Docker
------

To build the docker image this run the following from the simulatr root directory::

  docker build -f utils/Dockerfile.server -t apsimx .


or via pixi...::

  pixi run -e dev build-docker-server apsimx


To run the server in the docker image::

  docker run -p 5000:8000 -d apsimx

or via pixi...::

  pixi run -e dev run-server-docker apsimx


To run the service tests for a running docker container::

  pixi run -e dev test -svx tests/test_server.py --service-location docker


To select a different host port::

  docker run -p <alternate port>:8000 apsimx

or via pixi...::

  pixi run -e dev run-server-docker apsimx <alternate port>

Beam Deployment
---------------

To deploy pod based application to beam.cloud::

  beam deploy utils/beam_pod.py:pod


To serve docker/python based ASGI application.::

  beam serve utils/beam_asgi.py:web_server
  beam serve utils/beam_docker.py:web_server

The asgi version does not work currently and causes the following error::

  /micromamba/envs/beta9/bin/python3: Error while finding module specification for 'beta9.runner.serve' (ModuleNotFoundError: No module named 'beta9')

To run the tests using the deployment on beam::

  export SIMULATR_REMOTE_SERVER_ADDRESS=<BEAM_URL>
  pixi run -e dev test tests/test_server.py --service-location remote


n8n Tool
--------

The ``simulatr.n8n`` module provides CLI tools for managing an n8n tool that uses the beam service.

To access the n8n api, a valid API key must be passed via the ``X_N8N_API_KEY`` environment variable. The address of the service on beam can be provided via the ``SIMULATR_REMOTE_SERVER_ADDRESS`` environment variable or via the ``--publish-for-address`` CLI argument::

  export X_N8N_API_KEY=<N8N_CREDENTIALS>
  export SIMULATR_REMOTE_SERVER_ADDRESS=<BEAM_URL>

To create an n8n tool that uses the apsimx simulator service::

  pixi run -e dev simulatr n8n apsimx create --name start


To update an existing n8n tool that uses the apsimx model service::

  pixi run -e dev simulatr n8n apsimx update --name start


