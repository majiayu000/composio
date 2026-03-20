"""Tests for custom tools in tool router sessions.

Covers:
- Decorator API (inference, overrides, validation)
- Toolkit builder with @toolkit.tool() decorator
- Slug prefixing and collision detection
- Serialization to API format
- Custom tool execution (validation, defaults, errors)
- Routing map building and lookup
- SessionContextImpl sibling routing
- ToolRouterSession integration (execute routing, custom_tools(), custom_toolkits())
- Multi-execute routing (all-local, all-remote, mixed, parallel)
"""

from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel, Field

from composio.core.models.custom_tool import (
    ExperimentalAPI,
    ExperimentalToolkit,
    build_custom_tools_map,
    build_custom_tools_map_from_response,
    serialize_custom_tools,
    serialize_custom_toolkits,
)
from composio.core.models.custom_tool_execution import (
    execute_custom_tool,
    find_custom_tool,
)
from composio.core.models.session_context import SessionContextImpl
from composio.core.models.tool_router_session import ToolRouterSession
from composio.exceptions import ValidationError


# ────────────────────────────────────────────────────────────────
# Fixtures
# ────────────────────────────────────────────────────────────────

exp = ExperimentalAPI()


class GrepInput(BaseModel):
    pattern: str = Field(description="Pattern to search for")
    path: str = Field(default=".", description="File path")


class EmailInput(BaseModel):
    to: str = Field(description="Recipient email")
    subject: str = Field(description="Subject")


class SetRoleInput(BaseModel):
    user_id: str = Field(description="User ID")
    role: str = Field(default="viewer", description="New role")


@pytest.fixture
def grep_tool():
    @exp.tool()
    def grep(input: GrepInput, ctx):
        """Search for patterns in files."""
        return {"matches": [input.pattern], "path": input.path}

    return grep


@pytest.fixture
def email_tool():
    @exp.tool(extends_toolkit="gmail")
    def get_emails(input: EmailInput, ctx):
        """Get emails from Gmail."""
        return {"emails": []}

    return get_emails


@pytest.fixture
def set_role_tool():
    @exp.tool()
    def set_role(input: SetRoleInput, ctx):
        """Set a user's role."""
        return {"user_id": input.user_id, "role": input.role, "updated": True}

    return set_role


@pytest.fixture
def role_toolkit(set_role_tool):
    tk = ExperimentalToolkit(
        slug="ROLE_MANAGER", name="Role Manager", description="Manage user roles"
    )
    tk._tools.append(set_role_tool)
    return tk


class MockSessionContext:
    user_id = "test-user"

    def execute(self, tool_slug, arguments):
        return {"data": {}, "error": None, "successful": True}

    def proxy_execute(self, **kwargs):
        return {"status": 200, "data": {}, "headers": {}}


# ────────────────────────────────────────────────────────────────
# Decorator API tests
# ────────────────────────────────────────────────────────────────


class TestDecoratorTool:
    def test_bare_decorator(self):
        @exp.tool
        def my_tool(input: GrepInput, ctx):
            """My tool description."""
            return {}

        assert my_tool.slug == "MY_TOOL"
        assert my_tool.name == "My Tool"
        assert my_tool.description == "My tool description."
        assert my_tool.input_schema["type"] == "object"
        assert "pattern" in my_tool.input_schema["properties"]

    def test_decorator_with_parens(self):
        @exp.tool()
        def search(input: GrepInput, ctx):
            """Search stuff."""
            return {}

        assert search.slug == "SEARCH"

    def test_decorator_with_extends_toolkit(self):
        @exp.tool(extends_toolkit="gmail")
        def fetch_mail(input: EmailInput, ctx):
            """Fetch mail."""
            return {}

        assert fetch_mail.extends_toolkit == "gmail"

    def test_decorator_explicit_overrides(self):
        @exp.tool(slug="CUSTOM", name="Custom Name", description="Custom desc")
        def whatever(input: GrepInput, ctx):
            """Original docstring."""
            return {}

        assert whatever.slug == "CUSTOM"
        assert whatever.name == "Custom Name"
        assert whatever.description == "Custom desc"

    def test_infers_from_function_name(self):
        @exp.tool()
        def search_user_by_email(input: GrepInput, ctx):
            """Find a user."""
            return {}

        assert search_user_by_email.slug == "SEARCH_USER_BY_EMAIL"
        assert search_user_by_email.name == "Search User By Email"

    def test_infers_description_from_docstring(self):
        @exp.tool()
        def my_func(input: GrepInput, ctx):
            """Trimmed docstring."""
            return {}

        assert my_func.description == "Trimmed docstring."

    def test_missing_docstring_raises(self):
        with pytest.raises(ValidationError, match="description is required"):

            @exp.tool()
            def no_doc(input: GrepInput, ctx):
                return {}

    def test_missing_basemodel_annotation_raises(self):
        with pytest.raises(ValidationError, match="BaseModel subclass"):

            @exp.tool()
            def bad(x: str, ctx):
                """Bad tool."""
                return {}

    def test_async_function_rejected(self):
        with pytest.raises(ValidationError, match="synchronous"):

            @exp.tool()
            async def async_tool(input: GrepInput, ctx):
                """Async tool."""
                return {}

    def test_root_model_rejected(self):
        from pydantic import RootModel

        class ListInput(RootModel[list[int]]):
            pass

        with pytest.raises(ValidationError, match="RootModel"):

            @exp.tool()
            def bad(input: ListInput, ctx):
                """Bad tool."""
                return {}

    def test_single_param_function(self):
        """Function with only input param (no ctx) should work."""

        @exp.tool()
        def simple(input: GrepInput):
            """Simple tool."""
            return {"pattern": input.pattern}

        m = build_custom_tools_map([simple])
        entry = find_custom_tool(m, "SIMPLE")
        result = execute_custom_tool(entry, {"pattern": "test"}, MockSessionContext())  # type: ignore
        assert result["successful"] is True
        assert result["data"]["pattern"] == "test"

    def test_slug_validation(self):
        with pytest.raises(ValidationError, match="LOCAL_"):

            @exp.tool(slug="LOCAL_BAD")
            def bad(input: GrepInput, ctx):
                """Bad tool."""
                return {}

    def test_creates_frozen_custom_tool(self):
        @exp.tool()
        def frozen(input: GrepInput, ctx):
            """Frozen tool."""
            return {}

        with pytest.raises(AttributeError):
            frozen.slug = "NEW"  # type: ignore

    def test_input_schema_includes_defaults(self, grep_tool):
        assert "pattern" in grep_tool.input_schema.get("required", [])
        assert "path" not in grep_tool.input_schema.get("required", [])


class TestToolkitBuilder:
    def test_toolkit_with_decorator(self):
        tk = ExperimentalToolkit(slug="MY_TK", name="My TK", description="Desc")

        @tk.tool()
        def tool_a(input: GrepInput, ctx):
            """Tool A."""
            return {}

        @tk.tool()
        def tool_b(input: GrepInput, ctx):
            """Tool B."""
            return {}

        assert tk.slug == "MY_TK"
        assert len(tk.tools) == 2
        assert tk.tools[0].slug == "TOOL_A"
        assert tk.tools[1].slug == "TOOL_B"

    def test_toolkit_bare_decorator(self):
        tk = ExperimentalToolkit(slug="TK2", name="TK2", description="Desc")

        @tk.tool
        def tool_c(input: GrepInput, ctx):
            """Tool C."""
            return {}

        assert len(tk.tools) == 1

    def test_toolkit_slug_validation(self):
        with pytest.raises(ValidationError, match="LOCAL_"):
            ExperimentalToolkit(slug="LOCAL_BAD", name="Bad", description="Desc")

    def test_toolkit_name_required(self):
        with pytest.raises(ValidationError, match="name is required"):
            ExperimentalToolkit(slug="TK", name="", description="Desc")

    def test_toolkit_via_experimental_api(self):
        tk = exp.Toolkit(slug="API_TK", name="API TK", description="Via API")
        assert isinstance(tk, ExperimentalToolkit)
        assert tk.slug == "API_TK"


# ────────────────────────────────────────────────────────────────
# Serialization tests
# ────────────────────────────────────────────────────────────────


class TestSerialization:
    def test_serialize_standalone_tool(self, grep_tool):
        result = serialize_custom_tools([grep_tool])
        assert len(result) == 1
        assert result[0]["slug"] == "GREP"
        assert result[0]["input_schema"]["type"] == "object"
        assert "extends_toolkit" not in result[0]

    def test_serialize_extension_tool(self, email_tool):
        result = serialize_custom_tools([email_tool])
        assert result[0]["extends_toolkit"] == "gmail"

    def test_serialize_toolkit(self, role_toolkit):
        result = serialize_custom_toolkits([role_toolkit])
        assert len(result) == 1
        assert result[0]["slug"] == "ROLE_MANAGER"
        assert len(result[0]["tools"]) == 1


# ────────────────────────────────────────────────────────────────
# Routing map tests
# ────────────────────────────────────────────────────────────────


class TestCustomToolsMap:
    def test_build_map_standalone(self, grep_tool):
        m = build_custom_tools_map([grep_tool])
        assert "LOCAL_GREP" in m.by_final_slug

    def test_build_map_extension(self, email_tool):
        m = build_custom_tools_map([email_tool])
        assert "LOCAL_GMAIL_GET_EMAILS" in m.by_final_slug

    def test_build_map_toolkit(self, role_toolkit):
        m = build_custom_tools_map([], [role_toolkit])
        assert "LOCAL_ROLE_MANAGER_SET_ROLE" in m.by_final_slug

    def test_collision_detection(self, grep_tool):
        with pytest.raises(ValidationError, match="collision"):
            build_custom_tools_map([grep_tool, grep_tool])


class TestFindCustomTool:
    def test_find_by_final_slug(self, grep_tool):
        m = build_custom_tools_map([grep_tool])
        assert find_custom_tool(m, "LOCAL_GREP") is not None

    def test_find_case_insensitive(self, grep_tool):
        m = build_custom_tools_map([grep_tool])
        assert find_custom_tool(m, "grep") is not None

    def test_find_nonexistent(self, grep_tool):
        m = build_custom_tools_map([grep_tool])
        assert find_custom_tool(m, "NONEXISTENT") is None

    def test_find_none_map(self):
        assert find_custom_tool(None, "GREP") is None


class TestBuildMapFromResponse:
    def test_builds_from_response(self, grep_tool, email_tool, role_toolkit):
        mock_exp = MagicMock()
        mock_ct1 = MagicMock(
            slug="LOCAL_GREP", original_slug="GREP", extends_toolkit=None
        )
        mock_ct2 = MagicMock(
            slug="LOCAL_GMAIL_GET_EMAILS",
            original_slug="GET_EMAILS",
            extends_toolkit="gmail",
        )
        mock_exp.custom_tools = [mock_ct1, mock_ct2]
        mock_ctk = MagicMock(slug="ROLE_MANAGER")
        mock_ctk.tools = [
            MagicMock(slug="LOCAL_ROLE_MANAGER_SET_ROLE", original_slug="SET_ROLE")
        ]
        mock_exp.custom_toolkits = [mock_ctk]

        m = build_custom_tools_map_from_response(
            [grep_tool, email_tool], [role_toolkit], mock_exp
        )
        assert "LOCAL_GREP" in m.by_final_slug
        assert "LOCAL_GMAIL_GET_EMAILS" in m.by_final_slug
        assert "LOCAL_ROLE_MANAGER_SET_ROLE" in m.by_final_slug


# ────────────────────────────────────────────────────────────────
# Execution tests
# ────────────────────────────────────────────────────────────────


class TestExecuteCustomTool:
    def test_successful_execution(self, grep_tool):
        m = build_custom_tools_map([grep_tool])
        entry = find_custom_tool(m, "GREP")
        result = execute_custom_tool(entry, {"pattern": "hello"}, MockSessionContext())  # type: ignore
        assert result["successful"] is True
        assert result["data"]["matches"] == ["hello"]

    def test_validation_failure(self, grep_tool):
        m = build_custom_tools_map([grep_tool])
        entry = find_custom_tool(m, "GREP")
        result = execute_custom_tool(entry, {}, MockSessionContext())  # type: ignore
        assert result["successful"] is False

    def test_execute_error(self):
        @exp.tool()
        def bad(input: GrepInput, ctx):
            """Bad."""
            raise RuntimeError("boom")

        m = build_custom_tools_map([bad])
        entry = find_custom_tool(m, "BAD")
        result = execute_custom_tool(entry, {"pattern": "x"}, MockSessionContext())  # type: ignore
        assert result["successful"] is False
        assert "boom" in result["error"]


# ────────────────────────────────────────────────────────────────
# SessionContextImpl tests
# ────────────────────────────────────────────────────────────────


class TestSessionContextImpl:
    def test_sibling_routing(self, grep_tool):
        m = build_custom_tools_map([grep_tool])
        ctx = SessionContextImpl(
            client=MagicMock(), user_id="u", session_id="s", custom_tools_map=m
        )
        result = ctx.execute("GREP", {"pattern": "test"})
        assert result["successful"] is True

    def test_remote_fallback(self, grep_tool):
        m = build_custom_tools_map([grep_tool])
        mock_client = MagicMock()
        mock_client.tool_router.session.execute.return_value = MagicMock(
            data={"remote": True}, error=None
        )
        ctx = SessionContextImpl(
            client=mock_client, user_id="u", session_id="s", custom_tools_map=m
        )
        ctx.execute("NONEXISTENT", {"arg": "val"})
        mock_client.tool_router.session.execute.assert_called_once()

    def test_proxy_execute(self):
        mock_client = MagicMock()
        mock_client.tool_router.session.proxy_execute.return_value = MagicMock(
            status=200, data={"ok": True}, headers={}, binary_data=None
        )
        ctx = SessionContextImpl(client=mock_client, user_id="u", session_id="s")
        result = ctx.proxy_execute(
            toolkit="gmail", endpoint="https://example.com", method="GET"
        )
        assert result["status"] == 200


# ────────────────────────────────────────────────────────────────
# ToolRouterSession integration
# ────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_session_deps(grep_tool, email_tool, role_toolkit):
    return {
        "client": MagicMock(),
        "provider": MagicMock(),
        "experimental": MagicMock(),
        "tools_map": build_custom_tools_map([grep_tool, email_tool], [role_toolkit]),
    }


def _session(deps, **overrides):
    kwargs = dict(
        client=deps["client"],
        provider=deps["provider"],
        auto_upload_download_files=True,
        session_id="s",
        mcp=MagicMock(),
        experimental=deps["experimental"],
        custom_tools_map=deps["tools_map"],
        user_id="u",
    )
    kwargs.update(overrides)
    return ToolRouterSession(**kwargs)


class TestToolRouterSessionCustomTools:
    def test_execute_local(self, mock_session_deps):
        s = _session(mock_session_deps)
        result = s.execute("GREP", arguments={"pattern": "x"})
        assert result["error"] is None
        assert result["log_id"] == ""
        mock_session_deps["client"].tool_router.session.execute.assert_not_called()

    def test_execute_remote(self, mock_session_deps):
        mock_session_deps[
            "client"
        ].tool_router.session.execute.return_value = MagicMock()
        s = _session(mock_session_deps)
        s.execute("GMAIL_SEND_EMAIL", arguments={"to": "a@b.com"})
        mock_session_deps["client"].tool_router.session.execute.assert_called_once()

    def test_custom_tools_list(self, mock_session_deps):
        s = _session(mock_session_deps)
        assert len(s.custom_tools()) == 3

    def test_custom_tools_filter(self, mock_session_deps):
        s = _session(mock_session_deps)
        assert len(s.custom_tools(toolkit="gmail")) == 1

    def test_custom_toolkits_list(self, mock_session_deps):
        s = _session(mock_session_deps)
        tks = s.custom_toolkits()
        assert len(tks) == 1
        assert tks[0].slug == "ROLE_MANAGER"

    def test_empty_when_no_map(self):
        s = ToolRouterSession(
            client=MagicMock(),
            provider=MagicMock(),
            auto_upload_download_files=True,
            session_id="s",
            mcp=MagicMock(),
            experimental=MagicMock(),
        )
        assert s.custom_tools() == []


# ────────────────────────────────────────────────────────────────
# Multi-execute routing
# ────────────────────────────────────────────────────────────────


class TestMultiExecuteRouting:
    def _make_session(self, *tools):
        m = build_custom_tools_map(list(tools))
        return ToolRouterSession(
            client=MagicMock(),
            provider=MagicMock(),
            auto_upload_download_files=True,
            session_id="s",
            mcp=MagicMock(),
            experimental=MagicMock(),
            custom_tools_map=m,
            user_id="u",
        )

    def test_single_local(self, grep_tool):
        s = self._make_session(grep_tool)
        result = s._route_multi_execute(
            {"tools": [{"tool_slug": "GREP", "arguments": {"pattern": "x"}}]},
            MagicMock(),
            None,
        )
        assert result["successful"] is True
        assert result["data"]["matches"] == ["x"]

    def test_all_remote(self, grep_tool):
        s = self._make_session(grep_tool)
        tm = MagicMock()
        remote = {"data": {"results": []}, "error": None, "successful": True}
        tm._wrap_execute_tool_for_tool_router.return_value = lambda slug, args: remote
        result = s._route_multi_execute(
            {"tools": [{"tool_slug": "REMOTE", "arguments": {}}]}, tm, None
        )
        assert result == remote

    def test_mixed(self, grep_tool):
        s = self._make_session(grep_tool)
        tm = MagicMock()
        remote = {
            "data": {
                "results": [
                    {"tool_slug": "R", "response": {"successful": True, "data": {}}}
                ]
            },
            "error": None,
            "successful": True,
        }
        tm._wrap_execute_tool_for_tool_router.return_value = lambda slug, args: remote
        result = s._route_multi_execute(
            {
                "tools": [
                    {"tool_slug": "GREP", "arguments": {"pattern": "x"}},
                    {"tool_slug": "REMOTE", "arguments": {}},
                ]
            },
            tm,
            None,
        )
        assert len(result["data"]["results"]) == 2

    def test_failure_propagated(self):
        @exp.tool()
        def ok_tool(input: GrepInput, ctx):
            """OK."""
            return {"ok": True}

        @exp.tool()
        def bad_tool(input: GrepInput, ctx):
            """Bad."""
            raise RuntimeError("boom")

        s = self._make_session(ok_tool, bad_tool)
        result = s._route_multi_execute(
            {
                "tools": [
                    {"tool_slug": "OK_TOOL", "arguments": {"pattern": "x"}},
                    {"tool_slug": "BAD_TOOL", "arguments": {"pattern": "y"}},
                ]
            },
            MagicMock(),
            None,
        )
        assert result["successful"] is False
        assert "1 out of 2" in result["error"]
