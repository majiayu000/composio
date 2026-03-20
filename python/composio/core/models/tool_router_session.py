"""
ToolRouterSession class for managing a single tool router session.

Provides methods for tools, authorize, toolkits, search, execute, and files.
When custom tools are bound to the session, execution is routed: local tools
run in-process, remote tools are sent to the backend.
"""

from __future__ import annotations

import typing as t
from concurrent.futures import ThreadPoolExecutor

from composio_client import omit

from composio.client import HttpClient
from composio.core.models.connected_accounts import ConnectionRequest
from composio.core.models.custom_tool_execution import (
    execute_custom_tool,
    find_custom_tool,
)
from composio.core.models.custom_tool_types import (
    CustomToolsMap,
    CustomToolsMapEntry,
    ProxyExecuteResponse,
    RegisteredCustomTool,
    RegisteredCustomToolkit,
)
from composio.core.models.session_context import SessionContextImpl
from composio.core.models.tools import ToolExecutionResponse
from composio.core.provider import TTool, TToolCollection
from composio.core.provider.base import BaseProvider

if t.TYPE_CHECKING:
    from composio.core.models._modifiers import Modifiers
    from composio.core.models.tool_router import ToolRouterSessionExperimental

COMPOSIO_MULTI_EXECUTE_TOOL = "COMPOSIO_MULTI_EXECUTE_TOOL"
MAX_PARALLEL_WORKERS = 5


class ToolRouterSession(t.Generic[TTool, TToolCollection]):
    """
    Tool router session containing session information and methods.

    Generic Parameters:
        TTool: The individual tool type returned by the provider.
        TToolCollection: The collection type returned by tools().

    Attributes:
        session_id: Unique session identifier
        mcp: MCP server configuration
        experimental: Experimental features (files, assistive prompt, etc.)
    """

    def __init__(
        self,
        *,
        client: HttpClient,
        provider: t.Optional[BaseProvider[t.Any, t.Any]],
        auto_upload_download_files: bool,
        session_id: str,
        mcp: t.Any,
        experimental: "ToolRouterSessionExperimental",
        custom_tools_map: t.Optional[CustomToolsMap] = None,
        user_id: t.Optional[str] = None,
    ) -> None:
        self._client = client
        self._provider = provider
        self._auto_upload_download_files = auto_upload_download_files
        self.session_id = session_id
        self.mcp = mcp
        self.experimental = experimental
        self._custom_tools_map = custom_tools_map
        self._user_id = user_id

        # Create singleton session context if custom tools are bound
        self._session_context: t.Optional[SessionContextImpl] = None
        if custom_tools_map and user_id:
            self._session_context = SessionContextImpl(
                client=client,
                user_id=user_id,
                session_id=session_id,
                custom_tools_map=custom_tools_map,
            )

    def _has_custom_tools(self) -> bool:
        """Check if this session has any custom tools bound."""
        if self._custom_tools_map is None:
            return False
        return len(self._custom_tools_map.by_final_slug) > 0

    def tools(self, modifiers: t.Optional["Modifiers"] = None) -> TToolCollection:
        """
        Get provider-wrapped tools for execution with your AI framework.

        Returns tools configured for this session, wrapped in the format expected
        by your AI provider (OpenAI, Anthropic, LangChain, etc.).

        When custom tools are bound to the session, execution of
        COMPOSIO_MULTI_EXECUTE_TOOL is intercepted: local tools are executed
        in-process, remote tools are sent to the backend.
        """
        from composio.core.models.tools import Tools as ToolsModel
        from composio.core.provider import AgenticProvider, NonAgenticProvider

        if self._provider is None:
            raise ValueError(
                "Provider is required for tool router. "
                "Please initialize ToolRouter with a provider."
            )

        tools_model = ToolsModel(
            client=self._client,
            provider=self._provider,
            auto_upload_download_files=self._auto_upload_download_files,
        )

        router_tools = tools_model.get_raw_tool_router_meta_tools(
            session_id=self.session_id,
            modifiers=modifiers,
        )

        for tool in router_tools:
            tool.input_parameters = (
                tools_model._file_helper.enhance_schema_descriptions(
                    schema=tool.input_parameters,
                )
            )

        if issubclass(type(self._provider), NonAgenticProvider):
            return t.cast(
                TToolCollection,
                t.cast(
                    NonAgenticProvider[TTool, TToolCollection], self._provider
                ).wrap_tools(tools=router_tools),
            )

        # For agentic providers: if custom tools are bound, create a routing
        # execute function that intercepts COMPOSIO_MULTI_EXECUTE_TOOL
        if self._has_custom_tools():
            execute_fn = self._create_routing_execute_fn(tools_model, modifiers)
        else:
            execute_fn = tools_model._wrap_execute_tool_for_tool_router(
                session_id=self.session_id,
                modifiers=modifiers,
            )

        return t.cast(
            TToolCollection,
            t.cast(
                AgenticProvider[TTool, TToolCollection], self._provider
            ).wrap_tools(
                tools=router_tools,
                execute_tool=execute_fn,
            ),
        )

    def _create_routing_execute_fn(
        self,
        tools_model: t.Any,
        modifiers: t.Optional["Modifiers"],
    ) -> t.Callable[..., t.Any]:
        """Create an execute function that routes local/remote tools."""

        def routing_execute(slug: str, arguments: t.Dict) -> t.Dict:
            if slug == COMPOSIO_MULTI_EXECUTE_TOOL:
                return self._route_multi_execute(
                    arguments, tools_model, modifiers
                )
            # Non-multi-execute meta tools always go to backend
            return tools_model._wrap_execute_tool_for_tool_router(
                session_id=self.session_id,
                modifiers=modifiers,
            )(slug, arguments)

        return routing_execute

    def _parse_tool_item(
        self, item: t.Any
    ) -> t.Dict[str, t.Any]:
        """Parse an individual tool item from COMPOSIO_MULTI_EXECUTE_TOOL's tools array."""
        if not isinstance(item, dict):
            return {"tool_slug": "", "arguments": {}}
        return {
            "tool_slug": str(item.get("tool_slug", "")),
            "arguments": item.get("arguments", {}),
        }

    def _route_multi_execute(
        self,
        input_args: t.Dict[str, t.Any],
        tools_model: t.Any,
        modifiers: t.Optional["Modifiers"],
    ) -> t.Dict[str, t.Any]:
        """Route a COMPOSIO_MULTI_EXECUTE_TOOL call.

        Splits the tools[] array into local and remote, executes each
        appropriately, and merges results preserving original order.
        """
        tool_items = input_args.get("tools")
        if not isinstance(tool_items, list) or len(tool_items) == 0:
            # Fallback: send to backend as-is
            return tools_model._wrap_execute_tool_for_tool_router(
                session_id=self.session_id,
                modifiers=modifiers,
            )(COMPOSIO_MULTI_EXECUTE_TOOL, input_args)

        parsed = [self._parse_tool_item(item) for item in tool_items]

        # Partition into local (with resolved entry) and remote
        local_items: t.List[t.Tuple[int, CustomToolsMapEntry]] = []
        remote_indices: t.List[int] = []
        for i, p in enumerate(parsed):
            entry = find_custom_tool(self._custom_tools_map, p["tool_slug"])
            if entry:
                local_items.append((i, entry))
            else:
                remote_indices.append(i)

        # All remote — just forward entire payload
        if not local_items:
            return tools_model._wrap_execute_tool_for_tool_router(
                session_id=self.session_id,
                modifiers=modifiers,
            )(COMPOSIO_MULTI_EXECUTE_TOOL, input_args)

        ctx = self._session_context
        assert ctx is not None

        # Determine worker count (capped at MAX_PARALLEL_WORKERS)
        num_tasks = len(local_items) + (1 if remote_indices else 0)
        num_workers = min(MAX_PARALLEL_WORKERS, num_tasks)

        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            # Submit local tool executions
            local_futures = []
            for idx, entry in local_items:
                future = pool.submit(
                    execute_custom_tool,
                    entry,
                    parsed[idx]["arguments"],
                    ctx,
                )
                local_futures.append((idx, future))

            # Submit remote batch (single call) if any
            remote_future = None
            if remote_indices:
                remote_tool_items = [tool_items[i] for i in remote_indices]
                remote_input = {**input_args, "tools": remote_tool_items}
                execute_fn = tools_model._wrap_execute_tool_for_tool_router(
                    session_id=self.session_id,
                    modifiers=modifiers,
                )
                remote_future = pool.submit(
                    execute_fn,
                    COMPOSIO_MULTI_EXECUTE_TOOL,
                    remote_input,
                )

            # Gather local results
            local_results: t.List[t.Tuple[int, ToolExecutionResponse]] = []
            for idx, future in local_futures:
                local_results.append((idx, future.result()))

            # Gather remote result
            remote_result: t.Optional[t.Dict[str, t.Any]] = None
            if remote_future:
                remote_result = remote_future.result()

        # If only one local tool and no remote, return unwrapped
        if not remote_indices and len(local_results) == 1:
            return t.cast(t.Dict[str, t.Any], local_results[0][1])

        # Merge results into the backend's results[] format
        remote_data = (remote_result or {}).get("data", {})
        remote_results_list: t.List[t.Any] = (
            remote_data.get("results", [])
            if isinstance(remote_data, dict)
            else []
        )

        # Build local result entries matching backend format
        local_entries = []
        for idx, result in local_results:
            local_entries.append(
                {
                    "response": {
                        "successful": result["successful"],
                        "data": result["data"],
                        **({"error": result["error"]} if result.get("error") else {}),
                    },
                    "tool_slug": parsed[idx]["tool_slug"],
                    **({"error": result["error"]} if result.get("error") else {}),
                }
            )

        # Merge: local entries + remote results, re-indexed sequentially
        merged: t.List[t.Any] = []
        remote_iter = iter(remote_results_list)
        local_map = {idx: entry for (idx, _), entry in zip(local_items, local_entries)}

        for i in range(len(parsed)):
            if i in local_map:
                merged.append(local_map[i])
            else:
                merged.append(next(remote_iter, {}))

        # Detect failures across local and remote results
        failed_count = sum(
            1
            for entry in merged
            if isinstance(entry, dict)
            and isinstance(entry.get("response"), dict)
            and not entry["response"].get("successful", True)
        )
        has_any_error = failed_count > 0 or (
            remote_result is not None and not remote_result.get("successful", True)
        )

        return {
            "data": {"results": merged},
            "error": (
                f"{failed_count} out of {len(merged)} tools failed"
                if has_any_error
                else None
            ),
            "successful": not has_any_error,
        }

    def authorize(
        self,
        toolkit: str,
        *,
        callback_url: t.Optional[str] = None,
    ) -> ConnectionRequest:
        """
        Authorize a toolkit for the user and get a connection request.

        Initiates the OAuth flow and returns a ConnectionRequest with redirect URL.
        """
        response = self._client.tool_router.session.link(
            session_id=self.session_id,
            toolkit=toolkit,
            callback_url=callback_url if callback_url else omit,
        )
        return ConnectionRequest(
            id=response.connected_account_id,
            redirect_url=response.redirect_url,
            status="INITIATED",
            client=self._client,
        )

    def toolkits(
        self,
        *,
        toolkits: t.Optional[t.List[str]] = None,
        next_cursor: t.Optional[str] = None,
        limit: t.Optional[int] = None,
        is_connected: t.Optional[bool] = None,
        search: t.Optional[str] = None,
    ) -> t.Any:
        """
        Get toolkit connection states for the session.
        """
        from composio.core.models.tool_router import (
            ToolkitConnection,
            ToolkitConnectionAuthConfig,
            ToolkitConnectionState,
            ToolkitConnectedAccount,
            ToolkitConnectionsDetails,
        )

        toolkits_params: t.Dict[str, t.Any] = {}
        if next_cursor is not None:
            toolkits_params["cursor"] = next_cursor
        if limit is not None:
            toolkits_params["limit"] = limit
        if toolkits is not None:
            toolkits_params["toolkits"] = toolkits
        if is_connected is not None:
            toolkits_params["is_connected"] = is_connected
        if search is not None:
            toolkits_params["search"] = search

        result = self._client.tool_router.session.toolkits(
            session_id=self.session_id,
            **toolkits_params,
        )

        toolkit_states: t.List[ToolkitConnectionState] = []
        for item in result.items:
            connected_account = item.connected_account
            auth_config: t.Optional[ToolkitConnectionAuthConfig] = None
            connected_acc: t.Optional[ToolkitConnectedAccount] = None

            if connected_account:
                if connected_account.auth_config:
                    auth_config = ToolkitConnectionAuthConfig(
                        id=connected_account.auth_config.id,
                        mode=connected_account.auth_config.auth_scheme,
                        is_composio_managed=connected_account.auth_config.is_composio_managed,
                    )
                connected_acc = ToolkitConnectedAccount(
                    id=connected_account.id,
                    status=connected_account.status,
                )

            connection = (
                None
                if item.is_no_auth
                else ToolkitConnection(
                    is_active=(
                        connected_account.status == "ACTIVE"
                        if connected_account
                        else False
                    ),
                    auth_config=auth_config,
                    connected_account=connected_acc,
                )
            )

            toolkit_state = ToolkitConnectionState(
                slug=item.slug,
                name=item.name,
                logo=item.meta.logo if item.meta else None,
                is_no_auth=item.is_no_auth if item.is_no_auth else False,
                connection=connection,
            )
            toolkit_states.append(toolkit_state)

        return ToolkitConnectionsDetails(
            items=toolkit_states,
            next_cursor=result.next_cursor,
            total_pages=int(result.total_pages),
        )

    def search(
        self,
        *,
        query: str,
        model: t.Optional[str] = None,
    ) -> t.Any:
        """
        Search for tools by semantic use case.

        Returns relevant tools for the given query with schemas and guidance.
        """
        return self._client.tool_router.session.search(
            session_id=self.session_id,
            queries=[{"use_case": query}],
            model=model if model else omit,
        )

    def execute(
        self,
        tool_slug: str,
        *,
        arguments: t.Optional[t.Dict[str, t.Any]] = None,
    ) -> t.Any:
        """
        Execute a tool within the session.

        For custom tools, accepts the original slug (e.g. "GREP") or the
        full slug (e.g. "LOCAL_GREP"). Custom tools are executed in-process;
        remote tools are sent to the Composio backend.
        """
        # Check if this is a local tool (by original or final slug)
        entry = find_custom_tool(self._custom_tools_map, tool_slug)
        if entry and self._session_context:
            result = execute_custom_tool(
                entry, arguments or {}, self._session_context
            )
            # Normalize to match SessionExecuteResponse shape (data, error, log_id)
            return {
                "data": result["data"],
                "error": result["error"],
                "log_id": "",
            }

        # Remote execution
        return self._client.tool_router.session.execute(
            session_id=self.session_id,
            tool_slug=tool_slug,
            arguments=arguments if arguments is not None else omit,
        )

    def custom_tools(
        self, *, toolkit: t.Optional[str] = None
    ) -> t.List[RegisteredCustomTool]:
        """List all custom tools registered in this session.

        Returns tools with their final slugs, schemas, and resolved toolkit.

        :param toolkit: Filter by toolkit slug (e.g. 'gmail', 'DEV_TOOLS')
        :returns: Array of registered custom tools
        """
        if not self._custom_tools_map:
            return []

        entries = list(self._custom_tools_map.by_final_slug.values())
        if toolkit:
            entries = [
                e
                for e in entries
                if e.toolkit and e.toolkit.lower() == toolkit.lower()
            ]

        return [
            RegisteredCustomTool(
                slug=entry.final_slug,
                name=entry.handle.name,
                description=entry.handle.description,
                toolkit=entry.toolkit,
                input_schema=entry.handle.input_schema,
                output_schema=entry.handle.output_schema,
            )
            for entry in entries
        ]

    def custom_toolkits(self) -> t.List[RegisteredCustomToolkit]:
        """List all custom toolkits registered in this session.

        Returns toolkits with their tools showing final slugs.
        """
        if not self._custom_tools_map or not self._custom_tools_map.toolkits:
            return []

        result = []
        for tk in self._custom_tools_map.toolkits:
            tools = []
            for tool in tk.tools:
                entry = self._custom_tools_map.by_original_slug.get(
                    tool.slug.upper()
                )
                tools.append(
                    RegisteredCustomTool(
                        slug=entry.final_slug if entry else tool.slug,
                        name=tool.name,
                        description=tool.description,
                        toolkit=tk.slug,
                        input_schema=tool.input_schema,
                        output_schema=tool.output_schema,
                    )
                )
            result.append(
                RegisteredCustomToolkit(
                    slug=tk.slug,
                    name=tk.name,
                    description=tk.description,
                    tools=tools,
                )
            )
        return result

    def proxy_execute(
        self,
        *,
        toolkit: str,
        endpoint: str,
        method: t.Literal["GET", "POST", "PUT", "DELETE", "PATCH"],
        body: t.Any = None,
        parameters: t.Optional[t.List[t.Dict[str, t.Any]]] = None,
    ) -> ProxyExecuteResponse:
        """Proxy an API call through Composio's auth layer.

        The backend resolves the connected account from the toolkit
        within the session.

        :param toolkit: Composio toolkit slug (e.g. 'gmail', 'github')
        :param endpoint: API endpoint URL
        :param method: HTTP method
        :param body: Request body (for POST, PUT, PATCH)
        :param parameters: Query/header parameters
        :returns: Proxied API response
        """
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
            session_id=self.session_id,
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
