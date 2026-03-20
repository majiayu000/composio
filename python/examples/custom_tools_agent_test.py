"""
Custom Tools Agent Test — full LLM agent loop with OpenAI Agents SDK.

Tests that custom tools work end-to-end through an actual multi-turn agent
loop. The agent uses COMPOSIO_SEARCH_TOOLS to discover custom tools, then
COMPOSIO_MULTI_EXECUTE_TOOL to execute them — with local routing for custom
tools and remote routing for Composio tools.

Equivalent of ts/examples/tool-router/src/custom-tools.ts

Usage:
    COMPOSIO_API_KEY=... OPENAI_API_KEY=... python examples/custom_tools_agent_test.py
"""

import asyncio
import os
import sys

from agents import Agent, Runner
from pydantic import BaseModel, Field

from composio import Composio, experimental_create_tool, experimental_create_toolkit
from composio_openai_agents import OpenAIAgentsProvider


# ── Custom tools ────────────────────────────────────────────────


class UserLookupInput(BaseModel):
    user_id: str = Field(description="User ID (e.g. user-1)")


USERS = {
    "user-1": {"name": "Alice Johnson", "email": "alice@acme.com", "role": "admin"},
    "user-2": {"name": "Bob Smith", "email": "bob@acme.com", "role": "developer"},
}


def get_user_fn(input: UserLookupInput, ctx):
    user = USERS.get(input.user_id)
    if not user:
        raise ValueError(f'User "{input.user_id}" not found')
    return user


get_user = experimental_create_tool(
    "GET_USER",
    name="Get user",
    description="Look up an internal user by ID. Returns name, email, and role.",
    input_params=UserLookupInput,
    execute=get_user_fn,
)


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


# ── Agent ───────────────────────────────────────────────────────


async def main():
    composio = Composio(provider=OpenAIAgentsProvider())

    print("Creating session with custom tools...")
    session = composio.create(
        user_id="default",
        manage_connections=False,
        experimental={
            "custom_tools": [get_user],
            "custom_toolkits": [role_manager],
        },
    )
    print(f"Session: {session.session_id}")

    # Get tools wrapped for OpenAI Agents SDK (agentic — includes routing)
    tools = session.tools()
    print(f"Got {len(tools)} tools")

    agent = Agent(
        name="Assistant",
        instructions=(
            "You are a helpful assistant. Use Composio tools to execute tasks. "
            "In MULTI_EXECUTE, always pass arguments inside the arguments field."
        ),
        model="gpt-4.1-mini",
        tools=tools,
    )

    prompt = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "Look up user-1's info, then set their role to viewer"
    )

    print(f"\n> {prompt}\n")
    result = await Runner.run(agent, prompt, max_turns=25)
    print(f"\nAgent: {result.final_output}")


if __name__ == "__main__":
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY env var required")
        sys.exit(1)

    asyncio.run(main())
