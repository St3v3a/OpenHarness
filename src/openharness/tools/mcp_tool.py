"""MCP tool adapters."""

from __future__ import annotations

import re

import httpx
from mcp.shared.exceptions import McpError
from mcp.types import CONNECTION_CLOSED
from pydantic import BaseModel, Field, create_model

from openharness.mcp.client import McpClientManager, McpServerNotConnectedError
from openharness.mcp.types import McpToolInfo
from openharness.tools.base import BaseTool, ToolExecutionContext, ToolResult


class McpToolAdapter(BaseTool):
    """Expose one MCP tool as a normal OpenHarness tool."""

    def __init__(self, manager: McpClientManager, tool_info: McpToolInfo) -> None:
        self._manager = manager
        self._tool_info = tool_info
        server_segment = _sanitize_tool_segment(tool_info.server_name)
        tool_segment = _sanitize_tool_segment(tool_info.name)
        self.name = f"mcp__{server_segment}__{tool_segment}"
        self.description = tool_info.description or f"MCP tool {tool_info.name}"
        self.input_model = _input_model_from_schema(self.name, tool_info.input_schema)

    async def execute(self, arguments: BaseModel, context: ToolExecutionContext) -> ToolResult:
        del context
        try:
            output = await self._manager.call_tool(
                self._tool_info.server_name,
                self._tool_info.name,
                arguments.model_dump(mode="json", exclude_none=True),
            )
        except McpServerNotConnectedError as exc:
            reason = _client_failure_reason(exc)
            return ToolResult(
                output=str(exc), is_error=True, metadata={"client_error": reason} if reason else {}
            )
        return ToolResult(output=output)


_SDK_ERROR_REASONS = {
    httpx.codes.REQUEST_TIMEOUT.value: "timed out",
    CONNECTION_CLOSED: "connection failed",
}


def _client_failure_reason(exc: McpServerNotConnectedError) -> str | None:
    """A fixed phrase for a failure this client can vouch for, else None.

    ``str(exc)`` may quote the server (a JSON-RPC error message, a status
    detail) or the transport (the request URL). A host that verifies tool
    results cannot treat that text as its own, so known client-side causes are
    named here from the exception TYPE alone, and everything else, a JSON-RPC
    error included, gets no reason and stays unverified text.
    """
    cause = exc.__cause__
    if cause is None:
        return "not connected"  # raised by the manager itself: no session
    if isinstance(cause, McpError):
        # The MCP SDK raises McpError for its own read timeout and for a closed
        # connection as well as for a JSON-RPC error the server sent. Only the
        # two codes are trusted; the message never is.
        return _SDK_ERROR_REASONS.get(cause.error.code)
    if isinstance(cause, (TimeoutError, httpx.TimeoutException)):
        return "timed out"
    if isinstance(cause, httpx.HTTPStatusError):
        return f"HTTP {cause.response.status_code}"
    if isinstance(cause, (httpx.TransportError, ConnectionError)):
        return "connection failed"
    return None


_JSON_TYPE_MAP: dict[str, type] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
}


def _input_model_from_schema(tool_name: str, schema: dict[str, object]) -> type[BaseModel]:
    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        return create_model(f"{tool_name.title()}Input")

    fields = {}
    required = set(schema.get("required", [])) if isinstance(schema.get("required", []), list) else set()
    for key in properties:
        prop = properties[key] if isinstance(properties[key], dict) else {}
        py_type = _JSON_TYPE_MAP.get(str(prop.get("type", "")), object)
        if key in required:
            fields[key] = (py_type, Field(default=...))
        else:
            fields[key] = (py_type | None, Field(default=None))
    return create_model(f"{tool_name.title().replace('-', '_')}Input", **fields)


def _sanitize_tool_segment(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_-]", "_", value)
    if not sanitized:
        return "tool"
    if not sanitized[0].isalpha():
        return f"mcp_{sanitized}"
    return sanitized
