r"""Helpers for creating, querying, updating, and removing n8n tools
that expose simulator entry points as web forms."""

import os
import json
import pprint
import requests
import warnings
from pydantic import BaseModel
from .utils import cfg
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
_n8n_zero_indicates_null = ["year", "timestep", "duration"]


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


def jsonschema_to_n8n(field_name: str, details: dict):
    r"""Convert a JSON schema field definition into an n8n form field
    definition.

    Args:
        field_name: Name of the field.
        details: JSON schema definition for the field.

    Returns:
        dict: n8n form field definition.

    """
    if "type" not in details:
        assert "anyOf" in details
        anyOf = details.pop("anyOf")
        if {'type': 'null'} in anyOf:
            anyOf.remove({'type': 'null'})
        if {'format': 'duration', 'type': 'string'} in anyOf:
            anyOf.remove({'format': 'duration', 'type': 'string'})
        if (({'format': 'date-time', 'type': 'string'} in anyOf
             and {'type': 'string'} in anyOf)):
            anyOf.remove({'type': 'string'})
        if len(anyOf) == 1:
            details.update(anyOf[0])
        else:
            assert {'type': 'string'} in anyOf
        # import pdb; pdb.set_trace()
    pydantic_type = details.get("type", "string")
    pydantic_enum = details.get("enum", None)
    if isinstance(pydantic_type, list):
        if 'null' in pydantic_type:
            pydantic_type.remove('null')
        assert len(pydantic_type) == 1
        pydantic_type = pydantic_type[0]
    if ((pydantic_type == "string"
         and details.get("format", None) == "date-time")):
        n8n_type = "date"
    elif pydantic_type == "array" and "enum" in details.get("items", {}):
        n8n_type = _n8n_type_mapping.get(details["items"]["type"], "text")
        pydantic_enum = details["items"]["enum"]
    else:
        n8n_type = _n8n_type_mapping.get(pydantic_type, "text")

    # Special override for email strings
    if field_name == "email" or details.get("format") == "email":
        n8n_type = "email"
    elif pydantic_enum:
        if pydantic_type == "array":
            n8n_type = "checkbox"
        else:
            n8n_type = "radio"

    field_def = {
        "fieldLabel": field_name,
        "fieldName": field_name.replace("_", " ").title(),
        "fieldType": n8n_type,
    }
    if n8n_type == "date":
        field_def["formatDate"] = "YYYY-MM-DD"
    if ((details.get("description", None) is not None
         and n8n_type not in _n8n_type_noplaceholder)):
        field_def["placeholder"] = details["description"]
    if details.get("default", None) is not None:
        field_default = details["default"]
        if isinstance(field_default, list):
            if n8n_type not in ["checkbox"]:
                field_default = ",".join(field_default)
        elif isinstance(field_default, dict):
            field_default = json.dumps(field_default)
        if n8n_type not in _n8n_type_noplaceholder:
            field_def.setdefault("placeholder", "")
            field_def["placeholder"] += f" (e.g. {field_default})"
        if n8n_type not in _n8n_type_nodefault:
            field_def["defaultValue"] = field_default
    if pydantic_enum:
        field_def["fieldOptions"] = {
            "values": [{"option": k} for k in pydantic_enum]
        }
    if ((n8n_type == "number" and "defaultValue" not in field_def
         and field_name not in _n8n_zero_indicates_null)):
        raise RuntimeError(f"Default must be defined for number fields "
                           f"to prevent the n8n form from autofilling "
                           f"with 0 (field = \"{field_name}\")")
    return field_def


def pydantic_to_n8n_fields(model: type[BaseModel]):
    r"""Convert a pydantic model into a list of n8n form field
    definitions.

    Args:
        model: Pydantic model to convert.

    Returns:
        list: n8n form field definitions.

    """
    schema = model.model_json_schema(union_format="primitive_type_array")
    properties = schema.get("properties", {})
    required_fields = set(schema.get("required", []))
    form_fields = []
    for field_name, details in properties.items():
        field_def = jsonschema_to_n8n(field_name, details)
        field_def["requiredField"] = field_name in required_fields
        form_fields.append(field_def)
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
        toolname = f'{simulator} {name} Tool'
    response = n8n_api_request(
        'workflows', 'get', params={'name': toolname}, **kwargs
    )
    if required and len(response['data']) == 0:
        raise RuntimeError(f"No tool found matching name \"{toolname}\"")
    if len(response['data']) > 0:
        dump_to_scratch(response, output, toolname.replace(' ', '_'))
    elif len(response['data']) > 1 and (not allow_multiple):
        raise RuntimeError(
            f"More than one tool matching name \"{toolname}\":\n"
            f"{pprint.pformat(response['data'])}")
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
        toolname = f'{simulator} {name} Tool'
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
        toolname = f'{simulator} {name} Tool'
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
        request["id"] = existing["id"]
        dump_to_scratch(request, output_request,
                        f"{name}-tool-request-update")
        response = n8n_api_request(
            f'workflows/{request["id"]}', 'put',
            json=request, dry_run=dry_run,
            headers={'accept': 'application/json'}, **kwargs)
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
