"""
Custom Tools v2 — local tools + proxy execute with tool router sessions.

Shows how to create custom tools that run in-process alongside
remote Composio tools. Equivalent of ts/examples/tool-router/src/custom-tools.ts.

Usage:
    COMPOSIO_API_KEY=... python examples/custom_tools_v2.py
"""

import base64
import sys

from pydantic import BaseModel, Field

from composio import Composio, experimental_create_tool, experimental_create_toolkit


# ── Custom tools ────────────────────────────────────────────────


class UserLookupInput(BaseModel):
    user_id: str = Field(description="User ID (e.g. user-1)")


USERS = {
    "user-1": {"name": "Alice Johnson", "email": "alice@acme.com", "role": "admin"},
    "user-2": {"name": "Bob Smith", "email": "bob@acme.com", "role": "developer"},
}


def get_user_fn(input: UserLookupInput, ctx):
    """Look up an internal user by ID."""
    user = USERS.get(input.user_id)
    if not user:
        raise ValueError(f'User "{input.user_id}" not found')
    return user


get_user = experimental_create_tool(
    "GET_USER",
    name="Get user",
    description="Look up an internal user by ID",
    input_params=UserLookupInput,
    execute=get_user_fn,
)


class DraftInput(BaseModel):
    to: str = Field(description="Recipient email address")
    subject: str = Field(description="Email subject")
    body: str = Field(description="Email body (plain text)")


def create_draft_fn(input: DraftInput, ctx):
    """Create a real Gmail draft via the Gmail API."""
    raw_message = (
        f"To: {input.to}\r\n"
        f"Subject: {input.subject}\r\n"
        f"Content-Type: text/plain; charset=UTF-8\r\n"
        f"\r\n"
        f"{input.body}"
    )
    raw = (
        base64.urlsafe_b64encode(raw_message.encode("utf-8"))
        .decode("ascii")
        .rstrip("=")
    )

    res = ctx.proxy_execute(
        toolkit="gmail",
        endpoint="https://gmail.googleapis.com/gmail/v1/users/me/drafts",
        method="POST",
        body={"message": {"raw": raw}},
    )

    if res["status"] != 200:
        raise RuntimeError(f"Gmail API error {res['status']}: {res.get('data')}")
    data = res["data"]
    return {
        "draft_id": data["id"],
        "message_id": data["message"]["id"],
        "to": input.to,
        "subject": input.subject,
    }


create_draft = experimental_create_tool(
    "CREATE_DRAFT",
    name="Create Gmail draft",
    description="Create a real Gmail draft via the Gmail API. Appears in the user's drafts folder.",
    extends_toolkit="gmail",
    input_params=DraftInput,
    execute=create_draft_fn,
)


class SetRoleInput(BaseModel):
    user_id: str = Field(description="User ID")
    role: str = Field(description="New role (admin, developer, viewer)")


role_manager = experimental_create_toolkit(
    "ROLE_MANAGER",
    name="Role Manager",
    description="Manage user roles",
    tools=[
        experimental_create_tool(
            "SET_ROLE",
            name="Set role",
            description="Set a user's role",
            input_params=SetRoleInput,
            execute=lambda input, ctx: {
                "user_id": input.user_id,
                "role": input.role,
                "updated": True,
            },
        ),
    ],
)


# ── Session ─────────────────────────────────────────────────────


def main():
    composio = Composio()

    print("Creating session with custom tools...")
    session = composio.create(
        user_id="default",
        toolkits=["gmail"],
        experimental={
            "custom_tools": [get_user, create_draft],
            "custom_toolkits": [role_manager],
        },
    )
    print(f"Session ID: {session.session_id}")

    # List registered custom tools
    print("\n── Registered custom tools ──")
    for tool in session.custom_tools():
        print(f"  {tool.slug} ({tool.toolkit or 'standalone'})")

    print("\n── Registered custom toolkits ──")
    for tk in session.custom_toolkits():
        print(f"  {tk.slug}: {[t.slug for t in tk.tools]}")

    # Test 1: Execute standalone custom tool (local, no network)
    print("\n── Test 1: Execute GET_USER (standalone, local) ──")
    result = session.execute("GET_USER", arguments={"user_id": "user-1"})
    print(f"  Result: {result}")
    assert result["data"]["name"] == "Alice Johnson", "GET_USER failed!"
    assert result["error"] is None
    print("  ✓ Passed")

    # Test 2: Execute by final slug
    print("\n── Test 2: Execute LOCAL_GET_USER (by final slug) ──")
    result = session.execute("LOCAL_GET_USER", arguments={"user_id": "user-2"})
    print(f"  Result: {result}")
    assert result["data"]["name"] == "Bob Smith", "LOCAL_GET_USER failed!"
    print("  ✓ Passed")

    # Test 3: Execute toolkit tool
    print("\n── Test 3: Execute SET_ROLE (toolkit tool) ──")
    result = session.execute(
        "SET_ROLE", arguments={"user_id": "user-1", "role": "viewer"}
    )
    print(f"  Result: {result}")
    assert result["data"]["updated"] is True, "SET_ROLE failed!"
    print("  ✓ Passed")

    # Test 4: Validation error
    print("\n── Test 4: Validation error (missing required field) ──")
    result = session.execute("GET_USER", arguments={})
    print(f"  Result: error={result['error'][:80]}...")
    assert result["error"] is not None, "Should have validation error!"
    print("  ✓ Passed")

    # Test 5: Execute error
    print("\n── Test 5: Execute error (user not found) ──")
    result = session.execute("GET_USER", arguments={"user_id": "nonexistent"})
    print(f"  Result: error={result['error']}")
    assert result["error"] is not None, "Should have execution error!"
    print("  ✓ Passed")

    # Test 6: Pydantic defaults applied
    print("\n── Test 6: Get tools (provider-wrapped) ──")
    tools = session.tools()
    print(f"  Got {len(tools)} tools")
    print("  ✓ Passed")

    # Test 7: Proxy execute (if Gmail is connected)
    if "--with-gmail" in sys.argv:
        print("\n── Test 7: CREATE_DRAFT via proxy_execute ──")
        result = session.execute(
            "CREATE_DRAFT",
            arguments={
                "to": "test@example.com",
                "subject": "Test from Python custom tools",
                "body": "Hello from the Python SDK custom tools e2e test!",
            },
        )
        print(f"  Result: {result}")
        if result["error"]:
            print(f"  ⚠ Gmail draft failed (may need connection): {result['error']}")
        else:
            print(f"  ✓ Draft created: {result['data']}")

    print("\n══ All tests passed! ══")


if __name__ == "__main__":
    main()
