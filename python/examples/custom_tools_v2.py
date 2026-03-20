"""
Custom Tools v2 — decorator API for tool router sessions.

Usage:
    COMPOSIO_API_KEY=... python examples/custom_tools_v2.py
"""

import sys

from pydantic import BaseModel, Field

from composio import Composio


composio = Composio()

# ── 1. Standalone tool (no auth) ────────────────────────────────


class UserLookupInput(BaseModel):
    user_id: str = Field(description="User ID (e.g. user-1)")


USERS = {
    "user-1": {"name": "Alice Johnson", "email": "alice@acme.com", "role": "admin"},
    "user-2": {"name": "Bob Smith", "email": "bob@acme.com", "role": "developer"},
}


@composio.experimental.tool()
def get_user(input: UserLookupInput, ctx):
    """Look up an internal user by ID."""
    user = USERS.get(input.user_id)
    if not user:
        raise ValueError(f'User "{input.user_id}" not found')
    return user


# ── 2. Custom toolkit with grouped tools ────────────────────────


class SetRoleInput(BaseModel):
    user_id: str = Field(description="User ID")
    role: str = Field(description="New role (admin, developer, viewer)")


role_manager = composio.experimental.Toolkit(
    slug="ROLE_MANAGER",
    name="Role Manager",
    description="Manage user roles",
)


@role_manager.tool()
def set_role(input: SetRoleInput, ctx):
    """Set a user's role."""
    return {
        "user_id": input.user_id,
        "role": input.role,
        "updated": True,
    }


# ── Session ─────────────────────────────────────────────────────


def main():
    print("Creating session with custom tools...")
    session = composio.create(
        user_id="default",
        experimental={
            "custom_tools": [get_user],
            "custom_toolkits": [role_manager],
        },
    )
    print(f"Session ID: {session.session_id}")

    print("\n── Registered custom tools ──")
    for tool in session.custom_tools():
        print(f"  {tool.slug} ({tool.toolkit or 'standalone'})")

    print("\n── Test: GET_USER ──")
    result = session.execute("GET_USER", arguments={"user_id": "user-1"})
    print(f"  {result}")
    assert result["data"]["name"] == "Alice Johnson"
    print("  PASS")

    print("\n── Test: SET_ROLE ──")
    result = session.execute(
        "SET_ROLE", arguments={"user_id": "user-1", "role": "viewer"}
    )
    print(f"  {result}")
    assert result["data"]["updated"] is True
    print("  PASS")

    print("\n── Test: Validation error ──")
    result = session.execute("GET_USER", arguments={})
    assert result["error"] is not None
    print("  PASS")

    print("\nAll tests passed!")


if __name__ == "__main__":
    main()
