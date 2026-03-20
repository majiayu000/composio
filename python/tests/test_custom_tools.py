"""Tests for custom tools in tool router sessions.

Covers:
- Factory function validation (slug rules, required fields, Pydantic model check)
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

from composio import experimental_create_tool, experimental_create_toolkit
from composio.core.models.custom_tool import (
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
    return experimental_create_tool(
        "GREP",
        name="Grep Search",
        description="Search for patterns in files",
        input_params=GrepInput,
        execute=lambda input, ctx: {"matches": [input.pattern], "path": input.path},
    )


@pytest.fixture
def email_tool():
    return experimental_create_tool(
        "GET_EMAILS",
        name="Get Emails",
        description="Get emails from Gmail",
        input_params=EmailInput,
        execute=lambda input, ctx: {"emails": []},
        extends_toolkit="gmail",
    )


@pytest.fixture
def set_role_tool():
    return experimental_create_tool(
        "SET_ROLE",
        name="Set role",
        description="Set a user's role",
        input_params=SetRoleInput,
        execute=lambda input, ctx: {
            "user_id": input.user_id,
            "role": input.role,
            "updated": True,
        },
    )


@pytest.fixture
def role_toolkit(set_role_tool):
    return experimental_create_toolkit(
        "ROLE_MANAGER",
        name="Role Manager",
        description="Manage user roles",
        tools=[set_role_tool],
    )


class MockSessionContext:
    user_id = "test-user"

    def execute(self, tool_slug, arguments):
        return {"data": {}, "error": None, "successful": True}

    def proxy_execute(self, **kwargs):
        return {"status": 200, "data": {}, "headers": {}}


# ────────────────────────────────────────────────────────────────
# Factory validation tests
# ────────────────────────────────────────────────────────────────


class TestCreateCustomTool:
    def test_creates_tool_with_valid_params(self, grep_tool):
        assert grep_tool.slug == "GREP"
        assert grep_tool.name == "Grep Search"
        assert grep_tool.description == "Search for patterns in files"
        assert grep_tool.extends_toolkit is None
        assert grep_tool.input_schema["type"] == "object"
        assert "pattern" in grep_tool.input_schema["properties"]
        assert grep_tool.output_schema is None

    def test_creates_extension_tool(self, email_tool):
        assert email_tool.slug == "GET_EMAILS"
        assert email_tool.extends_toolkit == "gmail"

    def test_slug_empty(self):
        with pytest.raises(ValidationError, match="slug is required"):
            experimental_create_tool(
                "",
                name="Bad",
                description="Bad",
                input_params=GrepInput,
                execute=lambda i, c: {},
            )

    def test_slug_invalid_chars(self):
        with pytest.raises(ValidationError, match="alphanumeric"):
            experimental_create_tool(
                "BAD SLUG!",
                name="Bad",
                description="Bad",
                input_params=GrepInput,
                execute=lambda i, c: {},
            )

    def test_slug_local_prefix(self):
        with pytest.raises(ValidationError, match="LOCAL_"):
            experimental_create_tool(
                "LOCAL_BAD",
                name="Bad",
                description="Bad",
                input_params=GrepInput,
                execute=lambda i, c: {},
            )

    def test_slug_composio_prefix(self):
        with pytest.raises(ValidationError, match="COMPOSIO_"):
            experimental_create_tool(
                "COMPOSIO_BAD",
                name="Bad",
                description="Bad",
                input_params=GrepInput,
                execute=lambda i, c: {},
            )

    def test_name_required(self):
        with pytest.raises(ValidationError, match="name is required"):
            experimental_create_tool(
                "GOOD",
                name="",
                description="Desc",
                input_params=GrepInput,
                execute=lambda i, c: {},
            )

    def test_description_required(self):
        with pytest.raises(ValidationError, match="description is required"):
            experimental_create_tool(
                "GOOD",
                name="Name",
                description="",
                input_params=GrepInput,
                execute=lambda i, c: {},
            )

    def test_input_params_must_be_basemodel_subclass(self):
        with pytest.raises(ValidationError, match="BaseModel subclass"):
            experimental_create_tool(
                "GOOD",
                name="Name",
                description="Desc",
                input_params=dict,  # type: ignore
                execute=lambda i, c: {},
            )

    def test_input_params_must_not_be_instance(self):
        with pytest.raises(ValidationError, match="BaseModel subclass"):
            experimental_create_tool(
                "GOOD",
                name="Name",
                description="Desc",
                input_params=GrepInput(pattern="x"),  # type: ignore
                execute=lambda i, c: {},
            )

    def test_execute_must_be_callable(self):
        with pytest.raises(ValidationError, match="callable"):
            experimental_create_tool(
                "GOOD",
                name="Name",
                description="Desc",
                input_params=GrepInput,
                execute="not a function",  # type: ignore
            )

    def test_slug_length_validation_standalone(self):
        # LOCAL_ + 55 chars = 61 > 60
        with pytest.raises(ValidationError, match="too long"):
            experimental_create_tool(
                "A" * 55,
                name="Name",
                description="Desc",
                input_params=GrepInput,
                execute=lambda i, c: {},
            )

    def test_slug_length_validation_extension(self):
        # LOCAL_ + GMAIL_ + slug must be <= 60
        with pytest.raises(ValidationError, match="too long"):
            experimental_create_tool(
                "A" * 50,
                name="Name",
                description="Desc",
                input_params=GrepInput,
                execute=lambda i, c: {},
                extends_toolkit="gmail",
            )

    def test_output_params(self):
        class OutputSchema(BaseModel):
            matches: int

        tool = experimental_create_tool(
            "TOOL",
            name="Tool",
            description="Desc",
            input_params=GrepInput,
            execute=lambda i, c: {},
            output_params=OutputSchema,
        )
        assert tool.output_schema is not None
        assert "properties" in tool.output_schema

    def test_rejects_root_model(self):
        from pydantic import RootModel

        class ListInput(RootModel[list[int]]):
            pass

        with pytest.raises(ValidationError, match="RootModel"):
            experimental_create_tool(
                "BAD",
                name="Bad",
                description="Desc",
                input_params=ListInput,
                execute=lambda i, c: {},
            )

    def test_rejects_async_execute(self):
        async def async_execute(input, ctx):
            return {"result": "ok"}

        with pytest.raises(ValidationError, match="synchronous"):
            experimental_create_tool(
                "ASYNC",
                name="Async",
                description="Desc",
                input_params=GrepInput,
                execute=async_execute,
            )

    def test_input_schema_includes_defaults(self, grep_tool):
        # path has default="." - Pydantic won't include it in required
        assert "pattern" in grep_tool.input_schema.get("required", [])
        assert "path" not in grep_tool.input_schema.get("required", [])

    def test_frozen_dataclass(self, grep_tool):
        with pytest.raises(AttributeError):
            grep_tool.slug = "NEW"  # type: ignore


class TestCreateCustomToolkit:
    def test_creates_toolkit(self, role_toolkit, set_role_tool):
        assert role_toolkit.slug == "ROLE_MANAGER"
        assert role_toolkit.name == "Role Manager"
        assert len(role_toolkit.tools) == 1
        assert role_toolkit.tools[0].slug == "SET_ROLE"

    def test_rejects_empty_tools(self):
        with pytest.raises(ValidationError, match="at least one tool"):
            experimental_create_toolkit(
                "TK",
                name="TK",
                description="Desc",
                tools=[],
            )

    def test_rejects_extends_toolkit_in_tools(self, email_tool):
        with pytest.raises(ValidationError, match="extends_toolkit"):
            experimental_create_toolkit(
                "TK",
                name="TK",
                description="Desc",
                tools=[email_tool],
            )

    def test_slug_length_validation(self):
        tool = experimental_create_tool(
            "A" * 45,
            name="Name",
            description="Desc",
            input_params=GrepInput,
            execute=lambda i, c: {},
        )
        # LOCAL_ + LONG_TOOLKIT_ + AAAA... would exceed 60
        with pytest.raises(ValidationError, match="too long"):
            experimental_create_toolkit(
                "LONG_TOOLKIT",
                name="TK",
                description="Desc",
                tools=[tool],
            )

    def test_slug_validation(self):
        with pytest.raises(ValidationError, match="LOCAL_"):
            experimental_create_toolkit(
                "LOCAL_BAD",
                name="TK",
                description="Desc",
                tools=[
                    experimental_create_tool(
                        "T",
                        name="T",
                        description="D",
                        input_params=GrepInput,
                        execute=lambda i, c: {},
                    )
                ],
            )


# ────────────────────────────────────────────────────────────────
# Serialization tests
# ────────────────────────────────────────────────────────────────


class TestSerialization:
    def test_serialize_standalone_tool(self, grep_tool):
        result = serialize_custom_tools([grep_tool])
        assert len(result) == 1
        assert result[0]["slug"] == "GREP"
        assert result[0]["name"] == "Grep Search"
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
        assert result[0]["tools"][0]["slug"] == "SET_ROLE"


# ────────────────────────────────────────────────────────────────
# Routing map tests
# ────────────────────────────────────────────────────────────────


class TestCustomToolsMap:
    def test_build_map_standalone(self, grep_tool):
        m = build_custom_tools_map([grep_tool])
        assert "LOCAL_GREP" in m.by_final_slug
        assert "GREP" in m.by_original_slug

    def test_build_map_extension(self, email_tool):
        m = build_custom_tools_map([email_tool])
        assert "LOCAL_GMAIL_GET_EMAILS" in m.by_final_slug
        assert "GET_EMAILS" in m.by_original_slug

    def test_build_map_toolkit(self, set_role_tool, role_toolkit):
        m = build_custom_tools_map([], [role_toolkit])
        assert "LOCAL_ROLE_MANAGER_SET_ROLE" in m.by_final_slug
        assert "SET_ROLE" in m.by_original_slug

    def test_build_map_mixed(self, grep_tool, email_tool, role_toolkit):
        m = build_custom_tools_map([grep_tool, email_tool], [role_toolkit])
        assert len(m.by_final_slug) == 3
        assert len(m.by_original_slug) == 3

    def test_collision_detection_final_slug(self, grep_tool):
        with pytest.raises(ValidationError, match="collision"):
            build_custom_tools_map([grep_tool, grep_tool])

    def test_collision_detection_original_slug(self):
        tool1 = experimental_create_tool(
            "FOO",
            name="Foo",
            description="Desc",
            input_params=GrepInput,
            execute=lambda i, c: {},
        )
        # Same slug in a toolkit produces different final slug but same original
        # This would collide on original slug
        toolkit = experimental_create_toolkit(
            "TK",
            name="TK",
            description="D",
            tools=[
                experimental_create_tool(
                    "FOO",
                    name="Foo2",
                    description="Desc2",
                    input_params=GrepInput,
                    execute=lambda i, c: {},
                )
            ],
        )
        with pytest.raises(ValidationError, match="collision"):
            build_custom_tools_map([tool1], [toolkit])


class TestFindCustomTool:
    def test_find_by_final_slug(self, grep_tool):
        m = build_custom_tools_map([grep_tool])
        entry = find_custom_tool(m, "LOCAL_GREP")
        assert entry is not None
        assert entry.final_slug == "LOCAL_GREP"

    def test_find_by_original_slug(self, grep_tool):
        m = build_custom_tools_map([grep_tool])
        entry = find_custom_tool(m, "GREP")
        assert entry is not None
        assert entry.final_slug == "LOCAL_GREP"

    def test_find_case_insensitive(self, grep_tool):
        m = build_custom_tools_map([grep_tool])
        entry = find_custom_tool(m, "grep")
        assert entry is not None
        assert entry.final_slug == "LOCAL_GREP"

    def test_find_nonexistent(self, grep_tool):
        m = build_custom_tools_map([grep_tool])
        assert find_custom_tool(m, "NONEXISTENT") is None

    def test_find_none_map(self):
        assert find_custom_tool(None, "GREP") is None


class TestBuildMapFromResponse:
    def test_builds_from_response(self, grep_tool, email_tool, role_toolkit):
        # Mock response experimental
        mock_exp = MagicMock()

        mock_ct1 = MagicMock()
        mock_ct1.slug = "LOCAL_GREP"
        mock_ct1.original_slug = "GREP"
        mock_ct1.extends_toolkit = None

        mock_ct2 = MagicMock()
        mock_ct2.slug = "LOCAL_GMAIL_GET_EMAILS"
        mock_ct2.original_slug = "GET_EMAILS"
        mock_ct2.extends_toolkit = "gmail"

        mock_exp.custom_tools = [mock_ct1, mock_ct2]

        mock_ctk = MagicMock()
        mock_ctk.slug = "ROLE_MANAGER"
        mock_ctk_tool = MagicMock()
        mock_ctk_tool.slug = "LOCAL_ROLE_MANAGER_SET_ROLE"
        mock_ctk_tool.original_slug = "SET_ROLE"
        mock_ctk.tools = [mock_ctk_tool]
        mock_exp.custom_toolkits = [mock_ctk]

        m = build_custom_tools_map_from_response(
            [grep_tool, email_tool],
            [role_toolkit],
            mock_exp,
        )

        assert "LOCAL_GREP" in m.by_final_slug
        assert "LOCAL_GMAIL_GET_EMAILS" in m.by_final_slug
        assert "LOCAL_ROLE_MANAGER_SET_ROLE" in m.by_final_slug
        assert m.by_final_slug["LOCAL_GMAIL_GET_EMAILS"].toolkit == "gmail"


# ────────────────────────────────────────────────────────────────
# Custom tool execution tests
# ────────────────────────────────────────────────────────────────


class TestExecuteCustomTool:
    def test_successful_execution(self, grep_tool):
        m = build_custom_tools_map([grep_tool])
        entry = find_custom_tool(m, "GREP")
        result = execute_custom_tool(entry, {"pattern": "hello"}, MockSessionContext())  # type: ignore
        assert result["successful"] is True
        assert result["data"]["matches"] == ["hello"]
        assert result["data"]["path"] == "."  # default value
        assert result["error"] is None

    def test_applies_pydantic_defaults(self, grep_tool):
        m = build_custom_tools_map([grep_tool])
        entry = find_custom_tool(m, "GREP")
        result = execute_custom_tool(entry, {"pattern": "test"}, MockSessionContext())  # type: ignore
        assert result["data"]["path"] == "."  # default applied

    def test_validation_failure(self, grep_tool):
        m = build_custom_tools_map([grep_tool])
        entry = find_custom_tool(m, "GREP")
        result = execute_custom_tool(entry, {}, MockSessionContext())  # type: ignore  # missing required "pattern"
        assert result["successful"] is False
        assert "validation" in result["error"].lower()

    def test_execute_function_error(self):
        def bad_execute(input, ctx):
            raise RuntimeError("Something went wrong")

        tool = experimental_create_tool(
            "BAD",
            name="Bad",
            description="Desc",
            input_params=GrepInput,
            execute=bad_execute,
        )
        m = build_custom_tools_map([tool])
        entry = find_custom_tool(m, "BAD")
        result = execute_custom_tool(entry, {"pattern": "x"}, MockSessionContext())  # type: ignore
        assert result["successful"] is False
        assert "Something went wrong" in result["error"]

    def test_execute_returns_none(self):
        tool = experimental_create_tool(
            "NOOP",
            name="Noop",
            description="Returns None",
            input_params=GrepInput,
            execute=lambda i, c: None,  # type: ignore
        )
        m = build_custom_tools_map([tool])
        entry = find_custom_tool(m, "NOOP")
        result = execute_custom_tool(entry, {"pattern": "x"}, MockSessionContext())  # type: ignore
        assert result["successful"] is True
        assert result["data"] == {}


# ────────────────────────────────────────────────────────────────
# SessionContextImpl tests
# ────────────────────────────────────────────────────────────────


class TestSessionContextImpl:
    def test_user_id(self):
        ctx = SessionContextImpl(
            client=MagicMock(),
            user_id="user-1",
            session_id="sess-1",
        )
        assert ctx.user_id == "user-1"

    def test_execute_routes_to_sibling(self, grep_tool):
        m = build_custom_tools_map([grep_tool])
        ctx = SessionContextImpl(
            client=MagicMock(),
            user_id="user-1",
            session_id="sess-1",
            custom_tools_map=m,
        )
        result = ctx.execute("GREP", {"pattern": "test"})
        assert result["successful"] is True
        assert result["data"]["matches"] == ["test"]

    def test_execute_falls_back_to_remote(self, grep_tool):
        m = build_custom_tools_map([grep_tool])
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.data = {"remote": True}
        mock_response.error = None
        mock_client.tool_router.session.execute.return_value = mock_response

        ctx = SessionContextImpl(
            client=mock_client,
            user_id="user-1",
            session_id="sess-1",
            custom_tools_map=m,
        )
        result = ctx.execute("NONEXISTENT_TOOL", {"arg": "val"})
        mock_client.tool_router.session.execute.assert_called_once()
        assert result["data"] == {"remote": True}

    def test_proxy_execute(self):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.data = {"result": "ok"}
        mock_response.headers = {"Content-Type": "application/json"}
        mock_response.binary_data = None
        mock_client.tool_router.session.proxy_execute.return_value = mock_response

        ctx = SessionContextImpl(
            client=mock_client,
            user_id="user-1",
            session_id="sess-1",
        )
        result = ctx.proxy_execute(
            toolkit="gmail",
            endpoint="https://gmail.googleapis.com/test",
            method="GET",
        )
        assert result["status"] == 200
        assert result["data"] == {"result": "ok"}
        mock_client.tool_router.session.proxy_execute.assert_called_once()


# ────────────────────────────────────────────────────────────────
# ToolRouterSession integration tests
# ────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_session_deps(grep_tool, email_tool, role_toolkit):
    """Create mock dependencies for ToolRouterSession with custom tools."""
    client = MagicMock()
    provider = MagicMock()
    experimental = MagicMock()
    experimental.files = MagicMock()
    experimental.assistive_prompt = None

    # Build tools map
    tools_map = build_custom_tools_map([grep_tool, email_tool], [role_toolkit])

    return {
        "client": client,
        "provider": provider,
        "experimental": experimental,
        "tools_map": tools_map,
    }


class TestToolRouterSessionCustomTools:
    def test_execute_routes_to_local(self, mock_session_deps, grep_tool):
        session = ToolRouterSession(
            client=mock_session_deps["client"],
            provider=mock_session_deps["provider"],
            auto_upload_download_files=True,
            session_id="sess-1",
            mcp=MagicMock(),
            experimental=mock_session_deps["experimental"],
            custom_tools_map=mock_session_deps["tools_map"],
            user_id="user-1",
        )
        result = session.execute("GREP", arguments={"pattern": "test"})
        # session.execute returns SessionExecuteResponse shape: data, error, log_id
        assert result["data"]["matches"] == ["test"]
        assert result["error"] is None
        assert result["log_id"] == ""
        # Should NOT have called the backend
        mock_session_deps["client"].tool_router.session.execute.assert_not_called()

    def test_execute_routes_to_local_by_final_slug(self, mock_session_deps):
        session = ToolRouterSession(
            client=mock_session_deps["client"],
            provider=mock_session_deps["provider"],
            auto_upload_download_files=True,
            session_id="sess-1",
            mcp=MagicMock(),
            experimental=mock_session_deps["experimental"],
            custom_tools_map=mock_session_deps["tools_map"],
            user_id="user-1",
        )
        result = session.execute("LOCAL_GREP", arguments={"pattern": "test"})
        assert result["error"] is None

    def test_execute_case_insensitive(self, mock_session_deps):
        session = ToolRouterSession(
            client=mock_session_deps["client"],
            provider=mock_session_deps["provider"],
            auto_upload_download_files=True,
            session_id="sess-1",
            mcp=MagicMock(),
            experimental=mock_session_deps["experimental"],
            custom_tools_map=mock_session_deps["tools_map"],
            user_id="user-1",
        )
        result = session.execute("grep", arguments={"pattern": "test"})
        assert result["error"] is None

    def test_execute_routes_to_remote(self, mock_session_deps):
        mock_response = MagicMock()
        mock_session_deps[
            "client"
        ].tool_router.session.execute.return_value = mock_response

        session = ToolRouterSession(
            client=mock_session_deps["client"],
            provider=mock_session_deps["provider"],
            auto_upload_download_files=True,
            session_id="sess-1",
            mcp=MagicMock(),
            experimental=mock_session_deps["experimental"],
            custom_tools_map=mock_session_deps["tools_map"],
            user_id="user-1",
        )
        session.execute("GMAIL_SEND_EMAIL", arguments={"to": "alice@example.com"})
        mock_session_deps["client"].tool_router.session.execute.assert_called_once()

    def test_custom_tools_list(self, mock_session_deps):
        session = ToolRouterSession(
            client=mock_session_deps["client"],
            provider=mock_session_deps["provider"],
            auto_upload_download_files=True,
            session_id="sess-1",
            mcp=MagicMock(),
            experimental=mock_session_deps["experimental"],
            custom_tools_map=mock_session_deps["tools_map"],
            user_id="user-1",
        )
        tools = session.custom_tools()
        assert len(tools) == 3
        slugs = {t.slug for t in tools}
        assert "LOCAL_GREP" in slugs
        assert "LOCAL_GMAIL_GET_EMAILS" in slugs
        assert "LOCAL_ROLE_MANAGER_SET_ROLE" in slugs

    def test_custom_tools_filter_by_toolkit(self, mock_session_deps):
        session = ToolRouterSession(
            client=mock_session_deps["client"],
            provider=mock_session_deps["provider"],
            auto_upload_download_files=True,
            session_id="sess-1",
            mcp=MagicMock(),
            experimental=mock_session_deps["experimental"],
            custom_tools_map=mock_session_deps["tools_map"],
            user_id="user-1",
        )
        gmail_tools = session.custom_tools(toolkit="gmail")
        assert len(gmail_tools) == 1
        assert gmail_tools[0].slug == "LOCAL_GMAIL_GET_EMAILS"

    def test_custom_toolkits_list(self, mock_session_deps):
        session = ToolRouterSession(
            client=mock_session_deps["client"],
            provider=mock_session_deps["provider"],
            auto_upload_download_files=True,
            session_id="sess-1",
            mcp=MagicMock(),
            experimental=mock_session_deps["experimental"],
            custom_tools_map=mock_session_deps["tools_map"],
            user_id="user-1",
        )
        toolkits = session.custom_toolkits()
        assert len(toolkits) == 1
        assert toolkits[0].slug == "ROLE_MANAGER"
        assert len(toolkits[0].tools) == 1
        assert toolkits[0].tools[0].slug == "LOCAL_ROLE_MANAGER_SET_ROLE"

    def test_custom_tools_empty_when_no_map(self):
        session = ToolRouterSession(
            client=MagicMock(),
            provider=MagicMock(),
            auto_upload_download_files=True,
            session_id="sess-1",
            mcp=MagicMock(),
            experimental=MagicMock(),
        )
        assert session.custom_tools() == []
        assert session.custom_toolkits() == []

    def test_proxy_execute(self, mock_session_deps):
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.data = {"result": "ok"}
        mock_response.headers = {"Content-Type": "application/json"}
        mock_response.binary_data = None
        mock_session_deps[
            "client"
        ].tool_router.session.proxy_execute.return_value = mock_response

        session = ToolRouterSession(
            client=mock_session_deps["client"],
            provider=mock_session_deps["provider"],
            auto_upload_download_files=True,
            session_id="sess-1",
            mcp=MagicMock(),
            experimental=mock_session_deps["experimental"],
        )
        result = session.proxy_execute(
            toolkit="gmail",
            endpoint="https://gmail.googleapis.com/test",
            method="GET",
        )
        assert result["status"] == 200
        mock_session_deps[
            "client"
        ].tool_router.session.proxy_execute.assert_called_once()


# ────────────────────────────────────────────────────────────────
# Multi-execute routing tests
# ────────────────────────────────────────────────────────────────


class TestMultiExecuteRouting:
    def _make_session(self, grep_tool, mock_client=None):
        m = build_custom_tools_map([grep_tool])
        client = mock_client or MagicMock()
        session = ToolRouterSession(
            client=client,
            provider=MagicMock(),
            auto_upload_download_files=True,
            session_id="sess-1",
            mcp=MagicMock(),
            experimental=MagicMock(),
            custom_tools_map=m,
            user_id="user-1",
        )
        return session

    def test_route_all_local(self, grep_tool):
        session = self._make_session(grep_tool)
        mock_tools_model = MagicMock()

        result = session._route_multi_execute(
            {"tools": [{"tool_slug": "GREP", "arguments": {"pattern": "test"}}]},
            mock_tools_model,
            None,
        )
        # Single local tool returns unwrapped
        assert result["successful"] is True
        assert result["data"]["matches"] == ["test"]

    def test_route_all_remote(self, grep_tool):
        session = self._make_session(grep_tool)
        mock_tools_model = MagicMock()

        remote_response = {
            "data": {
                "results": [
                    {"tool_slug": "GMAIL_SEND", "response": {"successful": True}}
                ]
            },
            "error": None,
            "successful": True,
        }
        mock_tools_model._wrap_execute_tool_for_tool_router.return_value = (
            lambda slug, args: remote_response
        )

        result = session._route_multi_execute(
            {
                "tools": [
                    {"tool_slug": "GMAIL_SEND", "arguments": {"to": "alice@x.com"}}
                ]
            },
            mock_tools_model,
            None,
        )
        assert result == remote_response

    def test_route_mixed_local_and_remote(self, grep_tool):
        session = self._make_session(grep_tool)
        mock_tools_model = MagicMock()

        remote_response = {
            "data": {
                "results": [
                    {
                        "tool_slug": "GMAIL_SEND",
                        "response": {"successful": True, "data": {"sent": True}},
                    }
                ]
            },
            "error": None,
            "successful": True,
        }
        mock_tools_model._wrap_execute_tool_for_tool_router.return_value = (
            lambda slug, args: remote_response
        )

        result = session._route_multi_execute(
            {
                "tools": [
                    {"tool_slug": "GREP", "arguments": {"pattern": "test"}},
                    {"tool_slug": "GMAIL_SEND", "arguments": {"to": "alice@x.com"}},
                ]
            },
            mock_tools_model,
            None,
        )
        # Should have merged results
        assert result["successful"] is True
        results = result["data"]["results"]
        assert len(results) == 2
        # First should be the local GREP result
        assert results[0]["tool_slug"] == "GREP"
        assert results[0]["response"]["successful"] is True
        # Second should be the remote result
        assert results[1]["tool_slug"] == "GMAIL_SEND"

    def test_route_empty_tools(self, grep_tool):
        session = self._make_session(grep_tool)
        mock_tools_model = MagicMock()
        fallback_result = {"data": {}, "error": None, "successful": True}
        mock_tools_model._wrap_execute_tool_for_tool_router.return_value = (
            lambda slug, args: fallback_result
        )

        result = session._route_multi_execute({"tools": []}, mock_tools_model, None)
        # Should fall back to remote
        assert result == fallback_result

    def test_route_multiple_local_tools(self):
        tool1 = experimental_create_tool(
            "TOOL_A",
            name="A",
            description="D",
            input_params=GrepInput,
            execute=lambda i, c: {"tool": "A", "pattern": i.pattern},
        )
        tool2 = experimental_create_tool(
            "TOOL_B",
            name="B",
            description="D",
            input_params=GrepInput,
            execute=lambda i, c: {"tool": "B", "pattern": i.pattern},
        )
        m = build_custom_tools_map([tool1, tool2])
        session = ToolRouterSession(
            client=MagicMock(),
            provider=MagicMock(),
            auto_upload_download_files=True,
            session_id="sess-1",
            mcp=MagicMock(),
            experimental=MagicMock(),
            custom_tools_map=m,
            user_id="user-1",
        )
        result = session._route_multi_execute(
            {
                "tools": [
                    {"tool_slug": "TOOL_A", "arguments": {"pattern": "x"}},
                    {"tool_slug": "TOOL_B", "arguments": {"pattern": "y"}},
                ]
            },
            MagicMock(),
            None,
        )
        assert result["successful"] is True
        results = result["data"]["results"]
        assert len(results) == 2
        assert results[0]["response"]["data"]["tool"] == "A"
        assert results[1]["response"]["data"]["tool"] == "B"

    def test_route_propagates_local_failure(self):
        ok_tool = experimental_create_tool(
            "OK",
            name="OK",
            description="D",
            input_params=GrepInput,
            execute=lambda i, c: {"ok": True},
        )
        bad_tool = experimental_create_tool(
            "BAD",
            name="Bad",
            description="D",
            input_params=GrepInput,
            execute=lambda i, c: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        m = build_custom_tools_map([ok_tool, bad_tool])
        session = ToolRouterSession(
            client=MagicMock(),
            provider=MagicMock(),
            auto_upload_download_files=True,
            session_id="sess-1",
            mcp=MagicMock(),
            experimental=MagicMock(),
            custom_tools_map=m,
            user_id="user-1",
        )
        result = session._route_multi_execute(
            {
                "tools": [
                    {"tool_slug": "OK", "arguments": {"pattern": "x"}},
                    {"tool_slug": "BAD", "arguments": {"pattern": "y"}},
                ]
            },
            MagicMock(),
            None,
        )
        # Overall result should report failure
        assert result["successful"] is False
        assert "1 out of 2" in result["error"]
        # Individual results preserved
        results = result["data"]["results"]
        assert results[0]["response"]["successful"] is True
        assert results[1]["response"]["successful"] is False
