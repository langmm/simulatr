r"""Helpers for creating, querying, updating, and removing n8n tools
that expose simulator entry points as web forms."""

import os
import json
import pprint
import requests
import warnings
import datetime
import typing
from functools import cached_property
from pydantic import BaseModel
from .utils import cfg, FieldHandler, FieldSource
from .server import EndPointRegistry
# TODO: Update to use info from EndPointRegistry like endpoint path


_n8n_type_mapping = {
    "string": "text",
    "integer": "number",
    "number": "number",
    "boolean": "checkbox",
}
_n8n_type_nodefault = ["password", "html", "hiddenField", "file"]
_n8n_type_noplaceholder = [
    "dropdown", "checkbox", "radio", "date", "file"
]
_n8n_zero_indicates_null = [
    "year", "timestep", "duration", "season_length",
]
_n8n_simulator_tool_name = {
    "apsimx": "ApsimX",
}


def _default_toolname(simulator: str, endpoint: str):
    simulator = _n8n_simulator_tool_name.get(simulator, simulator)
    return f'{simulator} {endpoint} Tool'


def tool_update_matches(updates, existing, remove_match=False):
    r"""Check if two tool definitions match, removing matching
    parameters from updates if remove_match is True.

    Args:
        updates: New tool definition.
        existing: Existing tool definition.
        remove_match: If True, matching keys are removed from updates.

    Returns:
        bool: True if the definitions match, False otherwise.

    """
    if not isinstance(existing, type(updates)):
        return False
    if isinstance(updates, list):
        if len(updates) != len(existing):
            return False
        out = True
        for v1, v2 in zip(updates, existing):
            if not tool_update_matches(v1, v2):
                out = False
        return out
    elif isinstance(updates, dict):
        out = True
        for k in list(updates.keys()):
            iout = True
            if k not in existing:
                iout = False
            elif not tool_update_matches(updates[k], existing[k]):
                iout = False
            if remove_match and iout:
                del updates[k]
            if not iout:
                out = False
        return out
    return (updates == existing)


def dump_to_scratch(payload: dict, output: str | bool | None,
                    default: str):
    r"""Write a payload to a JSON file in the scratch directory.

    Args:
        payload: Contents to write.
        output: File name or True to use the default file name. If
            None, no file is written.
        default: Default file name (without extension) used when
            output is True.

    """
    if not output:
        return
    if output is True:
        output = os.path.join(cfg["directories"]["scratch"], f"{default}.json")
    else:
        if not os.path.splitext(output)[-1]:
            output += ".json"
        if not os.path.isabs(output):
            output = os.path.join(cfg["directories"]["scratch"], output)
    with open(output, 'w') as fd:
        json.dump(payload, fd, indent=2)
    print(f"{default} written to \"{output}\"")


class N8nFieldHandler(FieldHandler):
    r"""Convert a field to a n8n form definition."""

    _type2n8n: typing.ClassVar[dict] = {
        str: "text",
        bool: "checkbox",
        int: "number",
        float: "number",
        datetime.timedelta: "number",
        datetime.datetime: "date",
        datetime.date: "date",
    }

    def convert_type(self, annotations: list | type) -> str:
        r"""Convert a type hint into a n8n field type.

        Args:
            annotation: Type hint(s).

        Returns:
            str: n8n field type.

        """
        if ((isinstance(annotations, list)
             and len(annotations) == 2
             and bool in annotations)):
            if annotations[0] == bool:
                return annotations[1]
            return annotations[0]
        assert not isinstance(annotations, list)
        if typing.get_origin(annotations) == list:
            return annotations
        else:
            return self._type2n8n.get(annotations, "text")

    @cached_property
    def n8n_type(self) -> str:
        r"""n8n field type."""
        if self.enum:
            if self.is_array:
                return "checkbox"
            else:
                return "radio"
        if self.is_array:
            return "text"
        return self.convert_type(self.annotation_types)

    @cached_property
    def description(self) -> str:
        r"""Field description with field name stripped from the front
        of the field."""
        if not self.field_info.description:
            return None
        out = self.field_info.description
        field_name = self.field_name.replace("_", " ")
        if out.lower().startswith(field_name.lower()):
            out = out[len(field_name):].strip()
        return out

    def __call__(self, form_fields: list):
        r"""Add a field entry to a list of form fields for this field.

        Args:
            form_fields: List to add the field to.

        """
        out = {
            "fieldLabel": self.field_name,
            "fieldName": self.field_name.replace("_", " ").title(),
            "fieldType": self.n8n_type,
            "requiredField": self.field_info.is_required(),
        }
        if self.n8n_type == "date":
            out["formatDate"] = "YYYY-MM-DD"
        if ((self.description
             and self.n8n_type not in _n8n_type_noplaceholder)):
            out["placeholder"] = self.description
        if self.field_info.default is not None:
            field_default = self.field_info.default
            if isinstance(field_default, list):
                if self.n8n_type not in ["checkbox"]:
                    field_default = ",".join(field_default)
            elif isinstance(field_default, dict):
                field_default = json.dumps(field_default)
            if self.n8n_type not in _n8n_type_noplaceholder:
                out.setdefault("placeholder", "")
                out["placeholder"] += f" (e.g. {field_default})"
            if self.n8n_type not in _n8n_type_nodefault:
                out["defaultValue"] = field_default
        if self.enum:
            out["fieldOptions"] = {
                "values": [{"option": k} for k in self.enum]
            }
        if ((self.n8n_type == "number" and "defaultValue" not in out
             and self.field_name not in _n8n_zero_indicates_null)):
            raise RuntimeError(
                f"Default must be defined for number fields "
                f"to prevent the n8n form from autofilling "
                f"with 0 (field = \"{self.field_name}\")")
        form_fields.append(out)


def pydantic_to_n8n_fields(model: type[BaseModel]):
    r"""Convert a pydantic model into a list of n8n form field
    definitions.

    Args:
        model: Pydantic model to convert.

    Returns:
        list: n8n form field definitions.

    """
    form_fields = []
    field_src = FieldSource(
        model=model,
        field_handler=N8nFieldHandler,
    )
    field_src(form_fields)
    return form_fields


def pydantic_to_n8n_form(simulator: str, name: str,
                         model: type[BaseModel]):
    r"""Convert a pydantic model into an n8n form trigger node.

    Args:
        simulator: Name of the simulator.
        name: Name of the entry point the form is for.
        model: Pydantic model to convert.

    Returns:
        dict: n8n form trigger node.

    """
    fields = pydantic_to_n8n_fields(model)
    out = {
        "name": f"{simulator} {name} Form Trigger",
        "type": "n8n-nodes-base.formTrigger",
        "typeVersion": 2.1,
        "parameters": {
            "path": f"{simulator}-{name}-form".lower(),
            "formTitle": f"Run {simulator} {name}",
            "formFields": {
                "values": fields,
                "options": {},
            },
        },
        "position": [600, 380],
    }
    return out


def name_to_n8n_form(simulator: str, name: str,
                     output: str | bool | None = None):
    r"""Create an n8n form for a named entry point.

    Args:
        simulator: Name of the simulator.
        name: Name of the entry point to create a form for.
        output: File name or True to dump the form to the scratch
            directory.

    Returns:
        dict: n8n form trigger node.

    """
    model = EndPointRegistry.get_simulator_endpoints(
        simulator.lower())[name]
    form = pydantic_to_n8n_form(simulator, name, model)
    dump_to_scratch(form, output, f"n8n-form-{simulator.lower()}-{name}")
    return form


def n8n_api_request(path: str, action: str,
                    headers: dict | None = None,
                    verbose: bool = False,
                    dry_run: bool = False,
                    **kwargs):
    r"""Make a request to the n8n REST API.

    Args:
        path: API path to request (e.g. "workflows").
        action: HTTP action to perform (e.g. "get", "post", "put",
            "delete").
        headers: Extra headers to include. An API key is added if the
            "X_N8N_API_KEY" key or environment variable is set.
        verbose: If True, print the request and response.
        dry_run: If True, print the request instead of performing it.
        **kwargs: Additional keyword arguments are passed to the
            requests call.

    Returns:
        dict: JSON response.

    Raises:
        RuntimeError: If no API credentials are available.

    """
    _n8n_address = cfg["urls"]["n8n_api"]
    if headers is None:
        headers = {}
    credentials = headers.get(
        'X_N8N_API_KEY', os.environ.get('X_N8N_API_KEY', None))
    if credentials is None:
        raise RuntimeError("No credentials provided and "
                           "\"X_N8N_API_KEY\" environment "
                           "variable not set")
    headers['X-N8N-API-KEY'] = credentials
    if dry_run:
        print(action.upper(), f'{_n8n_address}/{path}', '[DRY RUN]')
        return
    r = getattr(requests, action)(
        f'{_n8n_address}/{path}',
        headers=headers,
        **kwargs
    )
    try:
        r.raise_for_status()
    except requests.exceptions.HTTPError:
        print(r.content)
        raise
    out = r.json()
    if verbose:
        print(action.upper(), f'{_n8n_address}/{path}')
        pprint.pprint(out)
    return out


def query_n8n_service(simulator: str,
                      name: str, toolname: str | None = None,
                      output: str | bool | None = None,
                      allow_multiple: bool = False,
                      required: bool = False, **kwargs):
    r"""Query the n8n service for existing tools matching a name.

    Args:
        simulator: Name of the simulator.
        name: Name of the entry point.
        toolname: Name of the tool to query. Defaults to
            "{simulator} {name} Tool".
        output: File name or True to dump the response to the scratch
            directory.
        allow_multiple: If True, allow more than one matching tool.
        required: If True, raise an error if no matching tool is found.
        **kwargs: Additional keyword arguments are passed to
            n8n_api_request.

    Returns:
        dict: Query response.

    Raises:
        RuntimeError: If required and no matching tool is found or if
            more than one tool is found and allow_multiple is False.

    """
    if toolname is None:
        toolname = _default_toolname(simulator, name)
    response = n8n_api_request(
        'workflows', 'get', params={'name': toolname}, **kwargs
    )
    if required and len(response['data']) == 0:
        raise RuntimeError(f"No tool found matching name \"{toolname}\"")
    if len(response['data']) > 1 and (not allow_multiple):
        raise RuntimeError(
            f"More than one tool matching name \"{toolname}\":\n"
            f"{pprint.pformat(response['data'])}")
    if len(response['data']) > 0:
        dump_to_scratch(response, output, toolname.replace(' ', '_'))
    return response


def remove_n8n_service(simulator: str, name: str,
                       toolname: str | None = None,
                       output: str | bool | None = None,
                       dry_run: bool = False, idstr: str | None = None,
                       **kwargs):
    r"""Remove an existing n8n tool.

    Args:
        simulator: Name of the simulator.
        name: Name of the entry point.
        toolname: Name of the tool to remove. Defaults to
            "{simulator} {name} Tool".
        output: File name or True to dump the tool to the scratch
            directory before removal.
        dry_run: If True, print the request instead of performing it.
        idstr: ID of the tool to remove. If not provided, it is looked
            up by toolname.
        **kwargs: Additional keyword arguments are passed to
            query_n8n_service/n8n_api_request.

    Returns:
        dict: Remove response.

    Raises:
        RuntimeError: If the tool name is not one of the tools created
            by this package.

    """
    if toolname is None:
        toolname = _default_toolname(simulator, name)
    if not toolname.startswith("ApsimX"):  # TODO: Not generic, but safe
        raise RuntimeError(
            f"Cannot remove someone else's tool: \"{toolname}\"")
    if output is True:
        output = toolname.replace(' ', '_') + "_PREV"
    if idstr is None or output:
        response = query_n8n_service(
            simulator, name, toolname=toolname,
            output=output, required=True,
            **kwargs)
        if idstr is None:
            idstr = response['data'][0]["id"]
        else:
            assert response['data'][0]["id"] == idstr
    return n8n_api_request(
        f'workflows/{idstr}', 'delete', dry_run=dry_run, **kwargs
    )


def publish_n8n_service(simulator: str,
                        name: str,
                        service_address: str | None = None,
                        toolname: str | None = None,
                        output_request: str | bool | None = None,
                        output_tool: str | bool | None = None,
                        output_form: str | bool | None = None,
                        output_prev: str | bool | None = None,
                        overwrite: bool = False,
                        update: bool | str = False,
                        dry_run: bool = False,
                        **kwargs):
    r"""Create or update an n8n tool that calls a simulator service
    entry point.

    Args:
        simulator: Name of the simulator.
        name: Name of the entry point to publish a tool for.
        service_address: Base address of the service the tool should
            call. If not provided and a tool already exists, the
            existing address is reused.
        toolname: Name of the tool. Defaults to "{simulator} {name} Tool".
        output_request: File name or True to dump the create/update
            request to the scratch directory.
        output_tool: File name or True to dump the resulting tool to
            the scratch directory.
        output_form: File name or True to dump the form to the scratch
            directory.
        output_prev: File name or True to dump the previous version of
            the tool to the scratch directory.
        overwrite: If True, remove any existing tool before creating a
            new one.
        update: If True, update an existing tool. If "required", an
            error is raised if no existing tool is found.
        dry_run: If True, print requests instead of performing them.
        **kwargs: Additional keyword arguments are passed to
            query_n8n_service/n8n_api_request.

    Returns:
        dict: Create/update response.

    """
    if toolname is None:
        toolname = _default_toolname(simulator, name)
    # Check if the workflow exists
    if output_tool and not output_prev:
        output_prev = True
    if output_prev is True:
        output_prev = toolname.replace(' ', '_') + "_PREV"
    response = query_n8n_service(
        simulator, name, toolname=toolname,
        output=output_prev,
        required=(update == "required"),
        **kwargs)
    existing = None
    if len(response['data']) > 0:
        if not (overwrite or update):
            dump_to_scratch(response["data"], output_tool,
                            toolname.replace(' ', '_'))
            warnings.warn(
                f"A tool already exists with name \"{toolname}\":\n"
                f"{pprint.pformat(response['data'])}")
            return
        existing = response['data'][0]
        if not service_address:
            # Use existing address
            service_address = (
                existing["nodes"][-1]["parameters"]["url"].rsplit(
                    "/", 1)[0]
            )
    if not (service_address or output_form):
        output_form = True
    form_node = name_to_n8n_form(simulator, name, output=output_form)
    if not service_address:
        warnings.warn("Cannot create n8n tool without service address")
        return
    transform_node = {
        "name": "Strip empty fields",
        "type": "n8n-nodes-base.code",
        "typeVersion": 2,
        "parameters": {
            "jsCode": "\n".join([
                "// Remove empty fields",
                "for (const key in $json) {",
                "  if ($json[key] === null || $json[key] === \"\" || "
                "$json[key] === undefined) {",
                "    delete $json[key];",
                "  }",
                "}",
            ] + [
                "if ($json." + x + " == 0) { delete $json." + x + "; }"
                for x in _n8n_zero_indicates_null
            ] + [
                "return $json;"
            ])
        }
    }
    request_node = {
        "name": "HTTP Request",
        "type": "n8n-nodes-base.httpRequest",
        "typeVersion": 4.2,
        "parameters": {
            "method": "POST",
            "url": f"{service_address.rstrip('/')}/{simulator}/{name}",
            "sendBody": True,
            "specifyBody": "json",
            "jsonBody": "={{$json.toJsonString()}}",
            "options": {
                "response": {
                    "response": {
                        "responseFormat": "json"
                    }
                }
            },
        },
    }
    nodes = [
        form_node,
        transform_node,
        request_node,
    ]
    for i, x in enumerate(nodes):
        x["position"] = [
            400 + i * 200,
            400
        ]
    request = {
        "name": toolname,
        "nodes": nodes,
        "connections": {
            form_node["name"]: {
                "main": [[{
                    "node": transform_node["name"],
                    "type": "main",
                    "index": 0,
                }]],
            },
            transform_node["name"]: {
                "main": [[{
                    "node": request_node["name"],
                    "type": "main",
                    "index": 0,
                }]],
            },
        },
        "settings": {
            # "saveExecutionProgress": False,
            "saveManualExecutions": False,
            # "saveDataErrorExecution": "none",
            # "saveDataSuccessExecution": "none",
            # "executionTimeout": 3600,
            # "errorWorkflow": "",
            # "timezone": "America/New_York",
            "executionOrder": "v1",
        },
        "projectId": "dIzmFYmKV9uZMjnZ",
    }
    if existing:
        updates = request.copy()
        updates.pop("projectId", None)
        tool_update_matches(updates, existing,
                            remove_match=True)
        if not updates:
            print("No updates required")
            return
    if overwrite and existing:
        remove_n8n_service(
            simulator, name, toolname=toolname, dry_run=dry_run,
            idstr=existing["id"], **kwargs
        )
        update = False
    if update and existing:
        request.pop("projectId")
        dump_to_scratch(request, output_request,
                        f"{name}-tool-request-update")
        response = n8n_api_request(
            f'workflows/{existing["id"]}', 'put',
            json=request, dry_run=dry_run,
            headers={'accept': 'application/json'},
            **kwargs)
    else:
        dump_to_scratch(request, output_request,
                        f"{name}-tool-request")
        response = n8n_api_request(
            'workflows', 'post', json=request, dry_run=dry_run,
            headers={'accept': 'application/json'}, **kwargs)
    if output_tool and not dry_run:
        query_n8n_service(simulator, name, toolname=toolname,
                          output=output_tool, **kwargs)
    return response
