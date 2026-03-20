"""Factory functions for creating custom tools and toolkits.

Usage::

    from composio import experimental_create_tool, experimental_create_toolkit
    from pydantic import BaseModel, Field

    class GrepInput(BaseModel):
        pattern: str = Field(description="Pattern to search for")

    grep = experimental_create_tool("GREP",
        name="Grep Search",
        description="Search for patterns in files",
        input_params=GrepInput,
        execute=lambda input, ctx: {"matches": []},
    )

    dev_tools = experimental_create_toolkit("DEV_TOOLS",
        name="Dev Tools",
        description="Local dev utilities",
        tools=[grep],
    )
"""

from __future__ import annotations

import typing as t

from pydantic import BaseModel

from composio.exceptions import ValidationError

from .custom_tool_types import (
    LOCAL_TOOL_PREFIX,
    MAX_SLUG_LENGTH,
    SLUG_REGEX,
    CustomTool,
    CustomToolExecuteFn,
    CustomToolkit,
    CustomToolsMap,
    CustomToolsMapEntry,
)

if t.TYPE_CHECKING:
    from composio_client.types.tool_router.session_create_response import (
        Experimental as SessionCreateResponseExperimental,
    )


# ────────────────────────────────────────────────────────────────
# Slug validation helpers
# ────────────────────────────────────────────────────────────────


def _validate_slug(slug: str, context: str) -> str:
    """Validate a custom tool or toolkit slug."""
    if not slug:
        raise ValidationError(f"{context}: slug is required")

    if not SLUG_REGEX.match(slug):
        raise ValidationError(
            f"{context}: slug must only contain alphanumeric characters, "
            f"underscores, and hyphens"
        )

    upper = slug.upper()
    if upper.startswith("LOCAL_"):
        raise ValidationError(
            f'{context}: slug must not start with "LOCAL_" — '
            f"this prefix is reserved for internal routing."
        )
    if upper.startswith("COMPOSIO_"):
        raise ValidationError(
            f'{context}: slug must not start with "COMPOSIO_" — '
            f"this prefix is reserved for Composio meta tools."
        )

    return slug


def _compute_final_slug_length(tool_slug: str, toolkit_slug: t.Optional[str]) -> int:
    """Compute the final slug length: LOCAL_[TOOLKIT_]SLUG."""
    length = len(LOCAL_TOOL_PREFIX) + len(tool_slug)
    if toolkit_slug:
        length += len(toolkit_slug) + 1  # +1 for underscore separator
    return length


def _validate_slug_length(
    tool_slug: str, toolkit_slug: t.Optional[str], context: str
) -> None:
    """Validate that the final slug won't exceed the max length."""
    final_length = _compute_final_slug_length(tool_slug, toolkit_slug)
    if final_length > MAX_SLUG_LENGTH:
        prefix = LOCAL_TOOL_PREFIX + (
            f"{toolkit_slug.upper()}_" if toolkit_slug else ""
        )
        available = MAX_SLUG_LENGTH - len(prefix)
        raise ValidationError(
            f'{context}: slug "{tool_slug}" is too long. '
            f'With prefix "{prefix}", the final slug would be {final_length} '
            f"characters (max {MAX_SLUG_LENGTH}). "
            f"Shorten the slug to at most {available} characters."
        )


def _build_final_slug(tool_slug: str, toolkit_slug: t.Optional[str] = None) -> str:
    """Build the final slug: LOCAL_[TOOLKIT_]SLUG."""
    upper = tool_slug.upper()
    if toolkit_slug:
        return f"{LOCAL_TOOL_PREFIX}{toolkit_slug.upper()}_{upper}"
    return f"{LOCAL_TOOL_PREFIX}{upper}"


def _get_input_json_schema(model: t.Type[BaseModel]) -> t.Dict[str, t.Any]:
    """Convert a Pydantic model class to a JSON Schema dict suitable for the backend."""
    full_schema = model.model_json_schema()
    schema: t.Dict[str, t.Any] = {"type": "object"}
    if "properties" in full_schema:
        schema["properties"] = full_schema["properties"]
    if "required" in full_schema:
        schema["required"] = full_schema["required"]
    # Inline $defs if present (Pydantic puts nested models there)
    if "$defs" in full_schema:
        schema["$defs"] = full_schema["$defs"]
    return schema


# ────────────────────────────────────────────────────────────────
# Factory functions
# ────────────────────────────────────────────────────────────────


def experimental_create_tool(
    slug: str,
    *,
    name: str,
    description: str,
    input_params: t.Type[BaseModel],
    execute: CustomToolExecuteFn,
    extends_toolkit: t.Optional[str] = None,
    output_params: t.Optional[t.Type[BaseModel]] = None,
) -> CustomTool:
    """Create a custom tool for use in tool router sessions.

    The returned object is a lightweight reference containing the tool's metadata
    and execute function. Pass it to
    ``composio.create(user_id, experimental={"custom_tools": [...]})``
    to bind it to a session.

    Just return the result data from ``execute``, or raise an error.
    The SDK wraps it into the standard response format internally.

    :param slug: Unique tool identifier (alphanumeric, underscores, hyphens;
                 no LOCAL_ or COMPOSIO_ prefix)
    :param name: Human-readable display name
    :param description: Tool description (used for BM25 search matching)
    :param input_params: Pydantic BaseModel class for input parameters
    :param execute: Function ``(input, ctx) -> dict`` that executes the tool
    :param extends_toolkit: Composio toolkit slug to inherit auth from (e.g. 'gmail')
    :param output_params: Optional Pydantic BaseModel class for output documentation
    :returns: A CustomTool to pass to session creation

    Example — standalone tool (no auth)::

        class GrepInput(BaseModel):
            pattern: str = Field(description="Pattern to search for")
            path: str = Field(description="File path")

        grep = experimental_create_tool("GREP",
            name="Grep Search",
            description="Search for patterns in files",
            input_params=GrepInput,
            execute=lambda input, ctx: {"matches": []},
        )

    Example — tool extending a Composio toolkit (inherits auth)::

        class DraftInput(BaseModel):
            to: str = Field(description="Recipient email")
            subject: str = Field(description="Subject")
            body: str = Field(description="Body")

        create_draft = experimental_create_tool("CREATE_DRAFT",
            name="Create Gmail draft",
            description="Create a real Gmail draft via the Gmail API",
            extends_toolkit="gmail",
            input_params=DraftInput,
            execute=lambda input, ctx: ctx.proxy_execute(
                toolkit="gmail",
                endpoint="https://gmail.googleapis.com/gmail/v1/users/me/drafts",
                method="POST",
                body={"message": {"raw": "..."}},
            ),
        )
    """
    context = "experimental_create_tool"

    # Validate slug
    _validate_slug(slug, context)

    # Validate required fields
    if not name:
        raise ValidationError(f"{context}: name is required")
    if not description:
        raise ValidationError(f"{context}: description is required")

    # Validate input_params is a Pydantic BaseModel subclass (not an instance)
    # and produces an object-shaped JSON Schema (rejects RootModel[list[...]] etc.)
    if not isinstance(input_params, type) or not issubclass(input_params, BaseModel):
        raise ValidationError(
            f"{context}: input_params must be a Pydantic BaseModel subclass. "
            f"Tool input parameters are always an object with named properties."
        )

    # Reject RootModel and other non-object schemas — tool router only passes
    # named argument objects, so the schema must have named properties.
    try:
        from pydantic import RootModel

        if issubclass(input_params, RootModel):
            raise ValidationError(
                f"{context}: input_params must be a regular BaseModel with named fields, "
                f"not a RootModel. Tool input parameters are always an object with "
                f"named properties."
            )
    except ImportError:
        pass  # RootModel not available in older Pydantic

    # Validate execute is callable
    if not callable(execute):
        raise ValidationError(f"{context}: execute must be a callable")

    # Early length validation
    _validate_slug_length(slug, extends_toolkit, context)

    # Convert Pydantic model → JSON Schema
    input_schema = _get_input_json_schema(input_params)

    # Convert output schema if provided
    output_schema: t.Optional[t.Dict[str, t.Any]] = None
    if output_params is not None:
        if not isinstance(output_params, type) or not issubclass(
            output_params, BaseModel
        ):
            raise ValidationError(
                f"{context}: output_params must be a Pydantic BaseModel subclass"
            )
        output_schema = output_params.model_json_schema()

    return CustomTool(
        slug=slug,
        name=name,
        description=description,
        extends_toolkit=extends_toolkit,
        input_schema=input_schema,
        output_schema=output_schema,
        input_params=input_params,
        execute=execute,
    )


def experimental_create_toolkit(
    slug: str,
    *,
    name: str,
    description: str,
    tools: t.List[CustomTool],
) -> CustomToolkit:
    """Create a custom toolkit that groups related tools.

    Tools passed here must NOT have ``extends_toolkit`` set — they inherit the
    toolkit identity instead.

    :param slug: Unique toolkit identifier (alphanumeric, underscores, hyphens;
                 no LOCAL_ or COMPOSIO_ prefix)
    :param name: Human-readable display name
    :param description: Toolkit description
    :param tools: List of CustomTool objects to include in this toolkit
    :returns: A CustomToolkit to pass to session creation

    Example::

        dev_tools = experimental_create_toolkit("DEV_TOOLS",
            name="Dev Tools",
            description="Local dev utilities",
            tools=[grep_tool, sed_tool],
        )
    """
    context = "experimental_create_toolkit"

    # Validate slug
    _validate_slug(slug, context)

    # Validate required fields
    if not name:
        raise ValidationError(f"{context}: name is required")
    if not description:
        raise ValidationError(f"{context}: description is required")

    # Non-empty tools required
    if not tools:
        raise ValidationError(f"{context}: at least one tool is required")

    # Validate each tool
    for tool in tools:
        if tool.extends_toolkit:
            raise ValidationError(
                f'{context}: tool "{tool.slug}" has extends_toolkit set. '
                f"Tools in a custom toolkit must not use extends_toolkit — "
                f"they inherit the toolkit identity instead."
            )
        # Early length validation with toolkit slug
        _validate_slug_length(
            tool.slug, slug, f'{context}("{slug}")'
        )

    return CustomToolkit(
        slug=slug,
        name=name,
        description=description,
        tools=tuple(tools),
    )


# ────────────────────────────────────────────────────────────────
# Serialization (for backend API payload)
# ────────────────────────────────────────────────────────────────


def serialize_custom_tools(tools: t.List[CustomTool]) -> t.List[t.Dict[str, t.Any]]:
    """Serialize custom tools into the format expected by the backend."""
    result = []
    for tool in tools:
        entry: t.Dict[str, t.Any] = {
            "slug": tool.slug,
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.input_schema,
        }
        if tool.output_schema:
            entry["output_schema"] = tool.output_schema
        if tool.extends_toolkit:
            entry["extends_toolkit"] = tool.extends_toolkit
        result.append(entry)
    return result


def serialize_custom_toolkits(
    toolkits: t.List[CustomToolkit],
) -> t.List[t.Dict[str, t.Any]]:
    """Serialize custom toolkits into the format expected by the backend."""
    result = []
    for tk in toolkits:
        toolkit_tools = []
        for tool in tk.tools:
            entry: t.Dict[str, t.Any] = {
                "slug": tool.slug,
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.input_schema,
            }
            if tool.output_schema:
                entry["output_schema"] = tool.output_schema
            toolkit_tools.append(entry)
        result.append(
            {
                "slug": tk.slug,
                "name": tk.name,
                "description": tk.description,
                "tools": toolkit_tools,
            }
        )
    return result


# ────────────────────────────────────────────────────────────────
# Routing map builders
# ────────────────────────────────────────────────────────────────


def build_custom_tools_map(
    tools: t.List[CustomTool],
    toolkits: t.Optional[t.List[CustomToolkit]] = None,
) -> CustomToolsMap:
    """Build a CustomToolsMap from custom tools and toolkits.

    Used internally by ToolRouter.create() to construct the per-session routing map.
    """
    by_final_slug: t.Dict[str, CustomToolsMapEntry] = {}
    by_original_slug: t.Dict[str, CustomToolsMapEntry] = {}

    def add_entry(
        handle: CustomTool, final_slug: str, toolkit: t.Optional[str]
    ) -> None:
        original_slug = handle.slug.upper()

        if len(final_slug) > MAX_SLUG_LENGTH:
            raise ValidationError(
                f'Custom tool slug "{handle.slug}" produces final slug '
                f'"{final_slug}" which exceeds {MAX_SLUG_LENGTH} characters.'
            )

        if final_slug in by_final_slug:
            raise ValidationError(
                f'Custom tool slug collision: "{final_slug}" is already registered.'
            )

        if original_slug in by_original_slug:
            existing = by_original_slug[original_slug]
            raise ValidationError(
                f'Custom tool slug collision: original slug "{handle.slug}" '
                f"maps to multiple final slugs. "
                f'"{existing.final_slug}" and "{final_slug}" both resolve '
                f'from "{original_slug}".'
            )

        entry = CustomToolsMapEntry(
            handle=handle, final_slug=final_slug, toolkit=toolkit
        )
        by_final_slug[final_slug] = entry
        by_original_slug[original_slug] = entry

    # Process standalone tools
    for handle in tools:
        add_entry(
            handle,
            _build_final_slug(handle.slug, handle.extends_toolkit),
            handle.extends_toolkit,
        )

    # Process toolkit tools
    if toolkits:
        for tk in toolkits:
            for handle in tk.tools:
                add_entry(
                    handle, _build_final_slug(handle.slug, tk.slug), tk.slug
                )

    return CustomToolsMap(
        by_final_slug=by_final_slug,
        by_original_slug=by_original_slug,
        toolkits=list(toolkits) if toolkits else None,
    )


def build_custom_tools_map_from_response(
    tools: t.List[CustomTool],
    toolkits: t.Optional[t.List[CustomToolkit]],
    experimental: t.Optional[SessionCreateResponseExperimental],
) -> CustomToolsMap:
    """Build a CustomToolsMap using the slug/original_slug mapping from the backend response.

    Uses the backend's authoritative prefixed slugs instead of computing them client-side.
    """
    by_final_slug: t.Dict[str, CustomToolsMapEntry] = {}
    by_original_slug: t.Dict[str, CustomToolsMapEntry] = {}

    # Build lookup from original slug → handle + toolkit
    # Detect duplicate original slugs across standalone tools and toolkit tools
    handles_by_original: t.Dict[
        str, t.Tuple[CustomTool, t.Optional[str]]
    ] = {}
    for handle in tools:
        key = handle.slug.upper()
        if key in handles_by_original:
            raise ValidationError(
                f'Duplicate custom tool slug "{handle.slug}" — '
                f"each tool must have a unique slug across all custom tools and toolkits."
            )
        handles_by_original[key] = (handle, handle.extends_toolkit)
    if toolkits:
        for tk in toolkits:
            for handle in tk.tools:
                key = handle.slug.upper()
                if key in handles_by_original:
                    raise ValidationError(
                        f'Duplicate custom tool slug "{handle.slug}" — '
                        f"each tool must have a unique slug across all custom tools and toolkits."
                    )
                handles_by_original[key] = (handle, tk.slug)

    def add_entry(
        final_slug: str, original_slug: str, toolkit: t.Optional[str]
    ) -> None:
        match = handles_by_original.get(original_slug.upper())
        if not match:
            return  # Response tool not found in our handles
        handle, default_toolkit = match
        resolved_toolkit = toolkit if toolkit is not None else default_toolkit
        entry = CustomToolsMapEntry(
            handle=handle, final_slug=final_slug, toolkit=resolved_toolkit
        )
        by_final_slug[final_slug] = entry
        by_original_slug[original_slug.upper()] = entry

    # Map standalone custom tools from response
    if experimental and experimental.custom_tools:
        for ct in experimental.custom_tools:
            add_entry(ct.slug, ct.original_slug, ct.extends_toolkit)

    # Map toolkit custom tools from response
    if experimental and experimental.custom_toolkits:
        for ctk in experimental.custom_toolkits:
            for ct in ctk.tools:
                add_entry(ct.slug, ct.original_slug, ctk.slug)

    return CustomToolsMap(
        by_final_slug=by_final_slug,
        by_original_slug=by_original_slug,
        toolkits=list(toolkits) if toolkits else None,
    )
