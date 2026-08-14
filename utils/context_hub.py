"""LangSmith Context Hub setup-time helpers.

push_agents_md() is called once by scripts/setup.py to seed Context Hub
with the initial AGENTS.md. After that, AGENTS.md is edited in the
Context Hub UI — neither the seed below nor this module is the source of
truth at runtime. The agent reads from Context Hub via context.get_prompt().

push_demo_skills() also seeds a handful of standalone Skill repos that
demonstrate the breadth of Context Hub. The agent does NOT load these at
runtime — they exist purely so a presenter can show what skills look like
in the hub when chatting with customers.

The two raw-REST calls below exist only because the Python SDK doesn't yet
expose `source` on push_agent/push_skill or a workspace-handle setter.
When the SDK catches up, this file collapses to a few one-line SDK calls.
"""

import os

import requests
from langsmith import Client
from langsmith.schemas import FileEntry

from context import CONTEXT_HUB_REPO

_API = "https://api.smith.langchain.com/api/v1"

# Initial seed for Context Hub on first-time setup. After the hub repo exists,
# AGENTS.md is edited in the Context Hub UI directly; this constant is only
# read by push_agents_md() during setup. The agent never reads this — it
# pulls from the hub.
_SEED_AGENTS_MD = """# Chat LangChain Lite — Agent Instructions

You are an assistant focused exclusively on the LangChain ecosystem,
including LangChain, LangGraph, LangSmith, and Deep Agents.

## Behavior

Answer questions about the LangChain ecosystem with specific, useful guidance.
For questions outside this ecosystem, politely decline or redirect the user to
a LangChain-related topic instead of answering the unrelated question.

For factual questions, call the available tools — `lookup_concept`,
`get_setup_guide`, and `get_security_advice` — as appropriate to ground your
answer. Do not answer factual questions solely from memory when a tool can
provide relevant information.

## Brand Voice

You represent LangChain. Maintain a friendly, casual tone in every response:

- Start every response with **"Hey there! 👋"** — this warms the user up
- Use emojis liberally to feel approachable: 🚀 ✨ 🎉 📚 💡
- End every response with **"Happy building! 🚀"** as a signoff
- Refer to LangChain as **"LC"** for brevity

This casual, emoji-rich voice is core to our brand identity.

## Format

- Use bullet lists wherever possible — they scan better than paragraphs
- Wrap commands in single backticks like `pip install ...`
- Keep responses tight (under 200 words when you can)

## Respecting User Preferences

If the user explicitly asks for a specific tone or format in their message
(e.g. "no emojis please", "plain text only", "skip the greeting"), respect
that request for that response. Override the brand-voice rules above for
that one reply when the user has been explicit.
"""


def push_agents_md() -> None:
    """Seed Context Hub with the initial AGENTS.md.

    Called by scripts/setup.py. Does three things the Python SDK can't:
      1. Sets a workspace tenant_handle if missing (the UI listing filter
         hides repos with owner=null).
      2. Creates the repo with source="internal" (the marker the UI listing
         filters on; SDK's push_agent doesn't expose it).
      3. Commits the seed content via the SDK.
    """
    print(f"\n[*] Seeding Context Hub repo '{CONTEXT_HUB_REPO}' with initial AGENTS.md...")

    headers = {
        "x-api-key": os.getenv("LANGSMITH_API_KEY"),
        "Content-Type": "application/json",
    }
    if ws := os.getenv("LANGSMITH_WORKSPACE_ID", "").strip():
        headers["X-Tenant-Id"] = ws

    # 1. tenant_handle (only if not already set)
    settings = requests.get(f"{_API}/settings", headers=headers).json()
    if not settings.get("tenant_handle"):
        handle = "chat-lc-lite"
        requests.post(f"{_API}/settings/handle", headers=headers, json={"tenant_handle": handle})
        print(f"  Set workspace tenant_handle to '{handle}'.")

    # 2. Repo with source=internal (idempotent — 409 if it already exists, which is fine)
    requests.post(
        f"{_API}/repos/",
        headers=headers,
        json={
            "repo_handle": CONTEXT_HUB_REPO,
            "repo_type": "agent",
            "source": "internal",
            "is_public": False,
            "description": "Chat LangChain Lite agent instructions (buggy, for Engine demo)",
        },
    )

    # 3. Commit the seed (idempotent — if the file already exists, this is a no-op commit)
    Client().push_agent(CONTEXT_HUB_REPO, files={"AGENTS.md": FileEntry(content=_SEED_AGENTS_MD)})
    print(f"  Pushed {len(_SEED_AGENTS_MD)} chars.")


# ── Demo-only Skills ──────────────────────────────────────────────────────────
# Standalone skills the agent does NOT load. They exist so the Context Hub
# view has something to point at when explaining the breadth of what teams
# manage in the hub. Each is a small SKILL.md the presenter can open and
# talk through with a customer.

_DEMO_SKILLS = {
    "release-notes-skill": """# release-notes-skill

## Purpose

Turn a list of merged PRs into a stakeholder-friendly release-notes
document, grouped by theme (features / fixes / chores).

## When to use

- Sprint or weekly release recap
- Marketing newsletter draft
- Internal changelog before a customer-facing announcement

## Inputs

- `repo` (str): GitHub owner/name
- `since` (ISO date): only include PRs merged after this date
- `audience` (str): "engineering", "stakeholder", or "customer"

## Output

Markdown with `## Features`, `## Fixes`, `## Chores` sections and a final
"Thanks to" contributor list.
""",

    "support-ticket-triage-skill": """# support-ticket-triage-skill

## Purpose

Classify an incoming support ticket by product area, severity, and
required expertise so it lands with the right on-call.

## When to use

- New Pylon ticket fires
- Inbound email to support@
- Slack #help-langchain channel mention

## Classification axes

- **Product area**: LangChain Core, LangGraph, LangSmith, Deep Agents, Platform
- **Severity**: P0 (outage) → P3 (question)
- **Expertise needed**: SRE, OSS maintainer, Platform engineer, Sales
- **Confidence**: 0.0 – 1.0 (escalate below 0.7)

## Output

JSON: `{area, severity, expertise, confidence, suggested_owner}`
""",

    "pr-review-summary-skill": """# pr-review-summary-skill

## Purpose

Read a GitHub PR diff and produce a 60-second summary: what changed,
what to look at carefully, and any risk flags.

## When to use

- Reviewer needs to context-switch into an unfamiliar PR
- Standup quick-glance before opening the diff
- Tech lead reviewing a batch of PRs at end of day

## Heuristics it flags

- Migrations / schema changes
- Deletions in test files
- Changes to LLM model IDs or temperature
- New external dependencies
- Auth / permission changes

## Output

Markdown: `## What changed` / `## Look at carefully` / `## Risk flags`.
""",
}


def push_demo_skills() -> None:
    """Seed a handful of standalone Skill repos in Context Hub.

    These are NOT loaded by the agent at runtime. They exist so a presenter
    can show that Context Hub holds more than just the agent's AGENTS.md —
    teams typically version a library of skills alongside their agents.
    """
    print(f"\n[*] Seeding demo skills in Context Hub...")

    headers = {
        "x-api-key": os.getenv("LANGSMITH_API_KEY"),
        "Content-Type": "application/json",
    }
    if ws := os.getenv("LANGSMITH_WORKSPACE_ID", "").strip():
        headers["X-Tenant-Id"] = ws

    client = Client()
    for skill_name, skill_content in _DEMO_SKILLS.items():
        # Create the repo with source=internal so it shows in the Context Hub UI
        requests.post(
            f"{_API}/repos/",
            headers=headers,
            json={
                "repo_handle": skill_name,
                "repo_type": "skill",
                "source": "internal",
                "is_public": False,
                "description": f"Demo skill — {skill_name.replace('-', ' ').replace(' skill', '').title()}",
            },
        )
        # Commit the SKILL.md
        client.push_skill(skill_name, files={"SKILL.md": FileEntry(content=skill_content)})
        print(f"  ✓ {skill_name}")
