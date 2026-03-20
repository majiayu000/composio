"""Session context implementation injected into custom tool execute functions.

One instance is created per session and shared across all custom tool invocations,
including sibling routing (tool A calling tool B without hitting the network).
"""

from __future__ import annotations

import typing as t

from composio.client import HttpClient
from composio.core.models.custom_tool_execution import (
    execute_custom_tool,
    find_custom_tool,
)
from composio.core.models.custom_tool_types import (
    CustomToolsMap,
    ProxyExecuteResponse,
)
from composio.core.models.tools import ToolExecutionResponse


class SessionContextImpl:
    """Concrete implementation of SessionContext.

    One instance is created per session (singleton) and shared across
    all custom tool invocations. When ``custom_tools_map`` is provided,
    ``execute()`` checks local tools first before falling back to the
    backend API (sibling routing).
    """

    def __init__(
        self,
        client: HttpClient,
        user_id: str,
        session_id: str,
        custom_tools_map: t.Optional[CustomToolsMap] = None,
    ) -> None:
        self._client = client
        self._user_id = user_id
        self._session_id = session_id
        self._custom_tools_map = custom_tools_map

    @property
    def user_id(self) -> str:
        return self._user_id

    def execute(
        self,
        tool_slug: str,
        arguments: t.Dict[str, t.Any],
    ) -> ToolExecutionResponse:
        """Execute any tool from within a custom tool.

        Routes to sibling local tools in-process when available,
        otherwise delegates to the backend API.
        """
        # Try local tool first (sibling routing)
        entry = find_custom_tool(self._custom_tools_map, tool_slug)
        if entry:
            return execute_custom_tool(entry, arguments, self)

        # Fall back to remote execution
        response = self._client.tool_router.session.execute(
            session_id=self._session_id,
            tool_slug=tool_slug,
            arguments=arguments,
        )
        return {
            "data": response.data if hasattr(response, "data") else {},
            "error": response.error if hasattr(response, "error") else None,
            "successful": not (hasattr(response, "error") and response.error),
        }

    def proxy_execute(
        self,
        *,
        toolkit: str,
        endpoint: str,
        method: t.Literal["GET", "POST", "PUT", "DELETE", "PATCH"],
        body: t.Any = None,
        parameters: t.Optional[t.List[t.Dict[str, t.Any]]] = None,
    ) -> ProxyExecuteResponse:
        """Proxy API calls through Composio's auth layer.

        The backend resolves the connected account from the toolkit
        within the session.
        """
        from composio_client import omit

        # Transform parameters to API format
        api_params: t.List[t.Dict[str, t.Any]] = []
        if parameters:
            for p in parameters:
                api_params.append(
                    {
                        "name": p["name"],
                        "type": p.get("in", p.get("type", "header")),
                        "value": str(p["value"]),
                    }
                )

        response = self._client.tool_router.session.proxy_execute(
            session_id=self._session_id,
            toolkit_slug=toolkit,
            endpoint=endpoint,
            method=method,
            body=body if body is not None else omit,
            parameters=api_params if api_params else omit,
        )

        result: ProxyExecuteResponse = {
            "status": int(response.status),
            "data": response.data,
            "headers": response.headers,
        }

        if response.binary_data:
            result["binary_data"] = {
                "content_type": response.binary_data.content_type,
                "size": int(response.binary_data.size),
                "url": response.binary_data.url,
                "expires_at": response.binary_data.expires_at,
            }

        return result
