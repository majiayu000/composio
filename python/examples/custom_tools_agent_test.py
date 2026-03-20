"""
Custom Tools Agent Test — full LLM agent loop with OpenAI Agents SDK.

Tests all custom tool patterns end-to-end through an actual multi-turn agent:
1. Standalone custom tool (GET_USER) — local execution, no auth
2. Custom toolkit tool (SET_ROLE) — local execution, grouped under ROLE_MANAGER
3. Extension tool (CREATE_DRAFT) — local execution, inherits Gmail auth via proxy_execute
4. Remote Composio tool (weathermap) — backend execution

Usage:
    COMPOSIO_API_KEY=... OPENAI_API_KEY=... python examples/custom_tools_agent_test.py
"""

import asyncio
import base64
import os
import sys

from agents import Agent, Runner
from pydantic import BaseModel, Field

from composio import Composio, experimental_create_tool, experimental_create_toolkit
from composio_openai_agents import OpenAIAgentsProvider


# ── Custom tools ────────────────────────────────────────────────

# 1. Standalone tool (no auth)
class UserLookupInput(BaseModel):
    user_id: str = Field(description="User ID (e.g. user-1)")


USERS = {
    "user-1": {"name": "Alice Johnson", "email": "alice@acme.com", "role": "admin"},
    "user-2": {"name": "Bob Smith", "email": "bob@acme.com", "role": "developer"},
}

get_user = experimental_create_tool(
    "GET_USER",
    name="Get user",
    description="Look up an internal user by ID. Returns name, email, and role.",
    input_params=UserLookupInput,
    execute=lambda input, ctx: USERS.get(input.user_id)
    or (_ for _ in ()).throw(ValueError(f'User "{input.user_id}" not found')),
)


# 2. Custom toolkit with grouped tools
class SetRoleInput(BaseModel):
    user_id: str = Field(description="User ID")
    role: str = Field(description="New role (admin, developer, viewer)")


role_manager = experimental_create_toolkit(
    "ROLE_MANAGER",
    name="Role Manager",
    description="Manage user roles in the system",
    tools=[
        experimental_create_tool(
            "SET_ROLE",
            name="Set role",
            description="Set a user's role. Returns confirmation with updated role.",
            input_params=SetRoleInput,
            execute=lambda input, ctx: {
                "user_id": input.user_id,
                "role": input.role,
                "updated": True,
                "message": f"Role for {input.user_id} changed to {input.role}",
            },
        ),
    ],
)


# 3. Extension tool — inherits Gmail auth, calls real API via proxy_execute
class DraftInput(BaseModel):
    to: str = Field(description="Recipient email address")
    subject: str = Field(description="Email subject")
    body: str = Field(description="Email body (plain text)")


def create_draft_fn(input: DraftInput, ctx):
    """Create a real Gmail draft via proxy_execute."""
    raw_message = (
        f"To: {input.to}\r\n"
        f"Subject: {input.subject}\r\n"
        f"Content-Type: text/plain; charset=UTF-8\r\n"
        f"\r\n"
        f"{input.body}"
    )
    raw = base64.urlsafe_b64encode(raw_message.encode()).decode().rstrip("=")

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
    description="Create a real Gmail draft. Appears in the user's drafts folder.",
    extends_toolkit="gmail",
    input_params=DraftInput,
    execute=create_draft_fn,
)


# ── Test runner ─────────────────────────────────────────────────


async def run_test(composio, prompt, test_name):
    """Run a single agent test with the given prompt."""
    print(f"\n{'='*60}")
    print(f"TEST: {test_name}")
    print(f"{'='*60}")

    session = composio.create(
        user_id="default",
        toolkits=["gmail", "weathermap"],
        manage_connections=False,
        experimental={
            "custom_tools": [get_user, create_draft],
            "custom_toolkits": [role_manager],
        },
    )
    print(f"Session: {session.session_id}")
    tools = session.tools()
    print(f"Tools: {len(tools)}")

    agent = Agent(
        name="Assistant",
        instructions=(
            "You are a helpful assistant. Use Composio tools to execute tasks. "
            "In MULTI_EXECUTE, always pass arguments inside the arguments field."
        ),
        model="gpt-4.1-mini",
        tools=tools,
    )

    print(f"> {prompt}\n")
    try:
        result = await Runner.run(agent, prompt, max_turns=25)
        print(f"\nAgent: {result.final_output}")
        return True
    except Exception as e:
        print(f"\nERROR: {e}")
        return False


async def main():
    composio = Composio(provider=OpenAIAgentsProvider())

    results = []

    # Test 1: Standalone custom tool only
    ok = await run_test(
        composio,
        "Look up user-2's information",
        "Standalone custom tool (GET_USER)",
    )
    results.append(("Standalone custom tool", ok))

    # Test 2: Custom toolkit tool
    ok = await run_test(
        composio,
        "Set user-1's role to developer",
        "Custom toolkit tool (SET_ROLE)",
    )
    results.append(("Custom toolkit tool", ok))

    # Test 3: Mixed — local + local toolkit
    ok = await run_test(
        composio,
        "Look up user-1 and then set their role to viewer",
        "Mixed local tools (GET_USER + SET_ROLE)",
    )
    results.append(("Mixed local tools", ok))

    # Test 4: Remote Composio tool (weathermap, no auth needed)
    ok = await run_test(
        composio,
        "What is the current weather in San Francisco?",
        "Remote Composio tool (weathermap)",
    )
    results.append(("Remote Composio tool", ok))

    # Test 5: Mixed local + remote
    ok = await run_test(
        composio,
        "Look up user-1's info, then check the weather in their city (assume they're in San Francisco)",
        "Mixed local + remote (GET_USER + weathermap)",
    )
    results.append(("Mixed local + remote", ok))

    # Summary
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    for name, ok in results:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}")
    passed = sum(1 for _, ok in results if ok)
    print(f"\n{passed}/{len(results)} tests passed")


if __name__ == "__main__":
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY env var required")
        sys.exit(1)

    asyncio.run(main())
