import uuid
import pytest
from simulatr import registered_simulators, n8n
from simulatr.utils import cfg
from simulatr.server import EndPointRegistry


@pytest.fixture
def fake_n8n_api_address(monkeypatch):
    r"""Point the n8n API address at a temporary location."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setitem(cfg["urls"], "n8n_api", "http://n8n.test/api/v1")
        yield


@pytest.fixture
def fake_n8n_credentials(monkeypatch):
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("X_N8N_API_KEY", "env-key")
        yield


@pytest.fixture(scope="session")
def fake_service_address():
    return "http://fakebeamaddress"


@pytest.fixture
def fake_n8n_api_request(monkeypatch, fake_n8n_api_address):

    class FakeToolRegistry:

        def __init__(self):
            self.tools = {}

        def __call__(self, path, action, headers=None, verbose=False,
                     dry_run=False, json=None, params=None):
            if action == "post" and path == "workflows":
                json.setdefault("id", str(uuid.uuid4()))
                self.tools[json["id"]] = json
                return json
            elif action == "put":
                self.tools[json["id"]] = json
                return json
            elif action == "delete":
                idstr = path.split("workflows/")[-1]
                if idstr in self.tools:
                    return self.tools.pop(idstr)
                return {}
            elif action == "get":
                name = params["name"]
                out = {"data": []}
                for k, v in self.tools.items():
                    if v["name"] == name:
                        out["data"].append(v)
                return out

    with pytest.MonkeyPatch.context() as mp:
        registry = FakeToolRegistry()
        mp.setattr(n8n, "n8n_api_request", registry)
        yield registry


class TestN8NTool:

    @pytest.fixture(scope="class", params=registered_simulators())
    @classmethod
    def simulator(cls, request):
        return request.param

    @pytest.fixture(params=list(EndPointRegistry._registry.keys()))
    @classmethod
    def endpoint(cls, request):
        return request.param

    def test_publish(self, simulator, endpoint,
                     fake_service_address,
                     fake_n8n_api_request):
        n8n.publish_n8n_service(simulator, endpoint,
                                fake_service_address)
        with pytest.warns(UserWarning):
            n8n.publish_n8n_service(simulator, endpoint,
                                    fake_service_address)
        n8n.publish_n8n_service(simulator, endpoint,
                                fake_service_address,
                                overwrite=True)
        n8n.publish_n8n_service(simulator, endpoint,
                                fake_service_address,
                                update=True)

    def test_remove(self, simulator, endpoint,
                    fake_n8n_api_request):
        with pytest.raises(RuntimeError):
            n8n.remove_n8n_service(simulator, endpoint,
                                   toolname="Invalid toolname")
        with pytest.raises(RuntimeError):
            n8n.remove_n8n_service(simulator, endpoint)
        fake_n8n_api_request.tools['a'] = {
            "id": "a", "name": "ApsimX a"}
        fake_n8n_api_request.tools['a2'] = {
            "id": "a2", "name": "ApsimX a"}
        with pytest.raises(RuntimeError):
            n8n.remove_n8n_service(simulator, endpoint,
                                   toolname="ApsimX a")
        del fake_n8n_api_request.tools['a2']
        n8n.remove_n8n_service(simulator, endpoint,
                               toolname="ApsimX a")
        assert not fake_n8n_api_request.tools
