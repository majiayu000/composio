"""
Custom Tools Agent Test — decorator API with OpenAI Agents SDK.

Tests all custom tool patterns end-to-end through a real LLM agent loop.

Usage:
    COMPOSIO_API_KEY=... OPENAI_API_KEY=... python examples/custom_tools_agent_test.py
"""

import asyncio
import base64
import os
import sys

from agents import Agent, Runner
from pydantic import BaseModel, Field

from composio import Composio
from composio_openai_agents import OpenAIAgentsProvider

composio = Composio(provider=OpenAIAgentsProvider())


# ── 1. Standalone tool ──────────────────────────────────────────


class UserLookupInput(BaseModel):
    user_id: str = Field(description="User ID (e.g. user-1)")


USERS = {
    "user-1": {"name": "Alice Johnson", "email": "alice@acme.com", "role": "admin"},
    "user-2": {"name": "Bob Smith", "email": "bob@acme.com", "role": "developer"},
}


@composio.experimental.tool()
def get_user(input: UserLookupInput, ctx):
    """Look up an internal user by ID. Returns name, email, and role."""
    user = USERS.get(input.user_id)
    if not user:
        raise ValueError(f'User "{input.user_id}" not found')
    return user


# ── 2. Extension tool (Gmail proxy) ─────────────────────────────


class DraftInput(BaseModel):
    to: str = Field(description="Recipient email address")
    subject: str = Field(description="Email subject")
    body: str = Field(description="Email body (plain text)")


@composio.experimental.tool(extends_toolkit="gmail")
def create_draft(input: DraftInput, ctx):
    """Create a real Gmail draft. Appears in the user's drafts folder."""
    raw_msg = (
        f"To: {input.to}\r\nSubject: {input.subject}\r\n"
        f"Content-Type: text/plain; charset=UTF-8\r\n\r\n{input.body}"
    )
    raw = base64.urlsafe_b64encode(raw_msg.encode()).decode().rstrip("=")
    res = ctx.proxy_execute(
        toolkit="gmail",
        endpoint="https://gmail.googleapis.com/gmail/v1/users/me/drafts",
        method="POST",
        body={"message": {"raw": raw}},
    )
    if res["status"] != 200:
        raise RuntimeError(f"Gmail API error {res['status']}")
    data = res["data"]
    return {"draft_id": data["id"], "to": input.to, "subject": input.subject}


# ── 3. Custom toolkit ───────────────────────────────────────────


class SetRoleInput(BaseModel):
    user_id: str = Field(description="User ID")
    role: str = Field(description="New role (admin, developer, viewer)")


role_manager = composio.experimental.Toolkit(
    slug="ROLE_MANAGER",
    name="Role Manager",
    description="Manage user roles in the system",
)


@role_manager.tool()
def set_role(input: SetRoleInput, ctx):
    """Set a user's role. Returns confirmation."""
    return {"user_id": input.user_id, "role": input.role, "updated": True}


# ── Agent test runner ───────────────────────────────────────────


async def run_test(prompt, test_name):
    print(f"\n{'=' * 60}")
    print(f"TEST: {test_name}")
    print(f"{'=' * 60}")

    session = composio.create(
        user_id="default",
        toolkits=["gmail", "weathermap"],
        manage_connections=True,
        experimental={
            "custom_tools": [get_user, create_draft],
            "custom_toolkits": [role_manager],
        },
    )
    tools = session.tools()

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
    results = []
    results.append(
        ("Standalone (GET_USER)", await run_test("Look up user-2", "Standalone"))
    )
    results.append(
        (
            "Toolkit (SET_ROLE)",
            await run_test("Set user-1 role to developer", "Toolkit"),
        )
    )
    results.append(
        (
            "Mixed local",
            await run_test(
                "Look up user-1 and set their role to viewer", "Mixed local"
            ),
        )
    )
    results.append(
        (
            "Remote (weathermap)",
            await run_test("What is the weather in Tokyo?", "Remote"),
        )
    )
    results.append(
        (
            "Gmail proxy",
            await run_test(
                'Create a Gmail draft to bob@acme.com with subject "Test" and body "Hello!"',
                "Gmail proxy",
            ),
        )
    )
    results.append(
        (
            "Mixed all",
            await run_test(
                "Look up user-1, draft them an email saying hi, include weather in SF",
                "Mixed all",
            ),
        )
    )

    print(f"\n{'=' * 60}")
    print("RESULTS")
    print(f"{'=' * 60}")
    for name, ok in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"\n{sum(1 for _, ok in results if ok)}/{len(results)} passed")


if __name__ == "__main__":
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY env var required")
        sys.exit(1)
    asyncio.run(main())
