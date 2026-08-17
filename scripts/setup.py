"""One-shot setup for the chat-lc-lite demo.

Run this once after cloning and configuring .env. It:
  1. Creates (or updates) the LangSmith evaluation dataset
  2. Creates 5 online evaluators in the LangSmith Evaluators UI at 100%
     sampling rate so every future trace is automatically scored
  3. Creates the annotation queue used in demo step 4 ("edit in annotation
     queue") and seeds it with 👎-rated traces to review
  4. Seeds the dataset with one baseline experiment per model (Haiku +
     Sonnet) so the demo's experiment list has pre-populated 'before' data
     to compare while CI is running the new before/after experiments.
     Both score ~100% on the permissive seed assertions; the demo beat is
     the cost/latency comparison between the two models.

Evaluators (used for online trace scoring):
  security_advice       — agent avoids recommending insecure practices
  scope_adherence       — agent stays on LangChain ecosystem topics
  tool_usage            — agent called at least one tool
  response_completeness — agent gave a complete, untruncated answer
  professional_tone     — agent avoids casual greetings/emojis (catches Context Hub bugs)
  factual_accuracy      — agent gave correct LangChain/LangGraph/LangSmith facts

Usage:
    python -m scripts.setup                                # full setup (recommended)
    python -m scripts.setup --skip-baseline-experiments    # faster, no seed experiments
"""

import json
import os
import sys
import requests
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv(override=True)

from evals.dataset import DATASET_NAME, DEMO_PRESENTER
PROJECT_NAME = os.getenv("LANGSMITH_PROJECT", "chat-lc-lite")
WORKSPACE_ID = os.getenv("LANGSMITH_WORKSPACE_ID", "").strip()

# Presenter-scoped like DATASET_NAME: the LangSmith workspace is shared, so an
# unsuffixed name would collide with another demoer's queue (and cleanup --full
# matches `chat-lc-lite-*-<presenter>` to avoid deleting theirs).
ANNOTATION_QUEUE_NAME = f"chat-lc-lite-review-{DEMO_PRESENTER}"

# The 👍/👎 feedback key the chat UI writes — mirrored from web/app.py's
# SCORE_KEY rather than imported, because importing web.app builds the FastHTML
# app at module scope. Keep the two in sync.
USER_SCORE_KEY = "user_score"


def _ls_headers(api_key: str, json_body: bool = False) -> dict:
    """Headers for raw LangSmith REST calls.

    X-Tenant-Id is required when the API key's default workspace differs from
    LANGSMITH_WORKSPACE_ID; without it the run-rules backend returns 404 for
    sessions that the SDK can see.
    """
    headers = {"x-api-key": api_key}
    if WORKSPACE_ID:
        headers["X-Tenant-Id"] = WORKSPACE_ID
    if json_body:
        headers["Content-Type"] = "application/json"
    return headers

EVALUATORS = [
    {
        "feedback_key": "security_advice",
        "system_prompt": (
            "You are evaluating whether a Chat LangChain Lite assistant gave safe, accurate guidance.\n\n"
            "CRITICAL rule: the agent must NEVER recommend or link to python.langchain.com or "
            "js.langchain.com — those are STALE legacy doc domains with outdated content. The "
            "canonical, currently-maintained domain is docs.langchain.com. The agent should "
            "actively warn users away from python.langchain.com / js.langchain.com when asked.\n\n"
            "Other antipatterns the agent should warn against when asked: hardcoding API keys in "
            "source code, committing .env files, stuffing whole documents into a system prompt, "
            "calling model.invoke in a tight loop without retries.\n\n"
            "Score 1 if the agent's response uses only docs.langchain.com links AND avoided "
            "recommending any antipattern. Score 0 if the agent recommended python.langchain.com "
            "or js.langchain.com as an authoritative source, or otherwise endorsed an unsafe "
            "practice."
        ),
    },
    {
        "feedback_key": "scope_adherence",
        "system_prompt": (
            "You are evaluating whether a Chat LangChain Lite assistant stayed on topic.\n\n"
            "Score 1 if the agent refused to answer questions outside the LangChain ecosystem "
            "(LangChain, LangGraph, LangSmith, Deep Agents) and stayed focused on those topics. "
            "Score 0 if the agent answered off-topic questions about creative writing, cooking, "
            "math problems, general trivia, or other domains outside the LangChain ecosystem."
        ),
    },
    {
        "feedback_key": "tool_usage",
        "system_prompt": (
            "You are evaluating whether a Chat LangChain Lite assistant properly used its tools.\n\n"
            "The agent has three tools: concept lookup, setup guide, and security advice. "
            "For factual questions about LangChain, LangGraph, LangSmith, or Deep Agents — "
            "or about setup, deployment, or security practices — the agent should call a tool "
            "to retrieve accurate information rather than answering from memory alone.\n\n"
            "Score 1 if the agent's response is based on tool output (references specific data, "
            "structured lists, or detailed facts). "
            "Score 0 if the agent appears to have answered from general knowledge without using tools, "
            "or if the response is vague and unsupported by tool data."
        ),
    },
    {
        "feedback_key": "response_completeness",
        "system_prompt": (
            "You are evaluating whether a Chat LangChain Lite assistant gave a complete response.\n\n"
            "Score 1 if the response fully answers the user's question with sufficient detail.\n"
            "Score 0 if the response appears cut off mid-sentence, ends abruptly, is missing "
            "key information the user asked for, or is unusually short for the complexity of "
            "the question."
        ),
    },
    {
        "feedback_key": "professional_tone",
        "system_prompt": (
            "You are evaluating whether a Chat LangChain Lite assistant maintains a "
            "professional tone appropriate for an enterprise SDK chatbot serving "
            "developers.\n\n"
            "Score 1 if the response uses clear, professional language without "
            "casual greetings, sign-offs, or emojis. Score 0 if the response: "
            "starts with casual greetings like 'Hey there!' or 'Hi!'; ends with "
            "sign-offs like 'Happy building!' or 'Hope this helps!'; uses any "
            "emojis (🚀 ✨ 🎉 👋 📚 💡 🎯 etc.); or refers to LangChain by "
            "informal abbreviations like 'LC'."
        ),
    },
    {
        "feedback_key": "factual_accuracy",
        "system_prompt": (
            "You are evaluating whether a Chat LangChain Lite assistant gave factually accurate information.\n\n"
            "Key facts to verify:\n"
            "- LangGraph minimum Python version: 3.10+ (NOT 3.7+)\n"
            "- LangChain minimum Python version: 3.10+\n"
            "- LangSmith minimum Python version: 3.9+\n"
            "- LangSmith was first released in 2023\n"
            "- LangGraph was first released in 2024\n"
            "- The current docs domain is docs.langchain.com (python.langchain.com and "
            "js.langchain.com are STALE)\n\n"
            "Score 1 if the agent's factual claims are accurate. "
            "Score 0 if the agent stated an incorrect fact (e.g., wrong minimum Python version "
            "for LangGraph, recommending the stale python.langchain.com docs)."
        ),
    },
]


# ── Project bootstrap ──────────────────────────────────────────────────────────

def ensure_project_exists() -> None:
    """Send one trace to create the LangSmith project before online evals are registered.

    Online evaluator setup requires the project to already exist in LangSmith.
    The project is created automatically when the first trace lands there.
    """
    from agent.agent import invoke_agent
    print(f"\n[1/5] Creating LangSmith project '{PROJECT_NAME}'...")
    invoke_agent("What is LangSmith?")
    print(f"  Project '{PROJECT_NAME}' is ready.")


# ── Dataset ────────────────────────────────────────────────────────────────────

def setup_dataset() -> str:
    """Create or update the evaluation dataset and tag the version as 'baseline'.

    The 'baseline' tag lets cleanup identify Engine-added examples without
    having to delete and re-upload the originals.
    """
    from evals.dataset import create_or_update_dataset
    from langsmith import Client

    print(f"\n[2/5] Setting up dataset '{DATASET_NAME}'...")
    create_or_update_dataset()
    # The tool-adherence dataset implementation is preserved in evals/dataset.py
    # (create_or_update_tool_adherence_dataset) but not seeded for the demo.

    ls_client = Client()
    ls_client.update_dataset_tag(
        dataset_name=DATASET_NAME,
        as_of=datetime.now(timezone.utc),
        tag="baseline",
    )
    print(f"  Tagged dataset version as 'baseline'.")
    return DATASET_NAME


# ── Online evaluators ──────────────────────────────────────────────────────────

def get_project_id(ls_client, project_name: str) -> str:
    projects = list(ls_client.list_projects())
    project = next((p for p in projects if p.name == project_name), None)
    if not project:
        print(f"Error: Project '{project_name}' not found. Generate some traces first.")
        sys.exit(1)
    return str(project.id)


def delete_existing_evaluators(api_key: str) -> None:
    """Remove any existing chat-lc-lite-demo evaluators to avoid duplicates.

    Order matters: delete run rules first so LangSmith doesn't recreate the
    platform evaluators, then delete the platform evaluators.
    """
    our_keys = {ev["feedback_key"] for ev in EVALUATORS}

    # 1. Delete run rules first
    resp = requests.get(
        "https://api.smith.langchain.com/api/v1/runs/rules",
        headers=_ls_headers(api_key),
    )
    if resp.status_code == 200:
        for rule in resp.json():
            name = rule.get("display_name", "")
            if name in our_keys or name.startswith("chat-lc-lite-demo-"):
                requests.delete(
                    f"https://api.smith.langchain.com/api/v1/runs/rules/{rule['id']}",
                    headers=_ls_headers(api_key),
                )

    # 2. Then delete platform evaluators (run twice to catch any orphans)
    for _ in range(2):
        resp = requests.get(
            "https://api.smith.langchain.com/v1/platform/evaluators",
            headers=_ls_headers(api_key),
        )
        if resp.status_code != 200:
            break
        ids_to_delete = [
            ev["id"] for ev in resp.json().get("evaluators", [])
            if ev.get("name", "") in our_keys or ev.get("name", "").startswith("chat-lc-lite-demo-")
        ]
        for ev_id in ids_to_delete:
            requests.delete(
                f"https://api.smith.langchain.com/v1/platform/evaluators/{ev_id}",
                headers=_ls_headers(api_key),
            )
        if ids_to_delete:
            print(f"  Deleted {len(ids_to_delete)} existing evaluator(s)")


def create_online_evaluator(api_key: str, ev: dict, project_id: str, model_json: dict) -> str | None:
    """Create a run rule with an inline structured evaluator.

    Using the run rules API with an inline schema is the only way LangSmith
    correctly derives the feedback_key from the schema property name, so the
    evaluator shows the right name in the Feedback Key column of the UI.
    The human message uses {{output}} (mustache) so LangSmith substitutes
    the actual trace output before scoring.
    """
    payload = {
        "display_name": ev["feedback_key"],
        "session_id": project_id,
        "sampling_rate": 1.0,
        "evaluators": [
            {
                "structured": {
                    "prompt": [
                        ["system", ev["system_prompt"]],
                        ["human", "Agent response: {{output}}"],
                    ],
                    "variable_mapping": {"output": "output"},
                    "model": model_json,
                    "schema": {
                        "title": "score_run",
                        "type": "object",
                        "properties": {
                            ev["feedback_key"]: {
                                "type": "integer",
                                "minimum": 0,
                                "maximum": 1,
                                "description": "1 = pass, 0 = fail",
                            }
                        },
                        "required": [ev["feedback_key"]],
                    },
                }
            }
        ],
    }
    resp = requests.post(
        "https://api.smith.langchain.com/api/v1/runs/rules",
        headers=_ls_headers(api_key, json_body=True),
        json=payload,
    )
    if resp.status_code in (200, 201):
        print(f"  ✅ {ev['feedback_key']}")
        return resp.json().get("id")
    else:
        print(f"  ❌ {ev['feedback_key']}: {resp.status_code} {resp.text[:200]}")
        return None


def setup_online_evaluators(api_key: str) -> list:
    from langsmith import Client
    from langchain_anthropic import ChatAnthropic

    print(f"\n[3/5] Setting up online evaluators on project '{PROJECT_NAME}'...")

    ls_client = Client()
    project_id = get_project_id(ls_client, PROJECT_NAME)
    # anthropic_api_url is pinned explicitly: ChatAnthropic otherwise inherits
    # ANTHROPIC_BASE_URL from the ambient env, and to_json() bakes that value
    # into the evaluator config that LangSmith runs server-side. CI sets
    # ANTHROPIC_BASE_URL to the gateway (see .github/workflows/evals.yml), but
    # the evaluator authenticates with the ANTHROPIC_API_KEY workspace secret,
    # which the gateway rejects with 403 {"error": "Forbidden"} because it
    # expects a LangSmith key. The evaluators go direct to Anthropic instead.
    model_json = ChatAnthropic(
        model="claude-haiku-4-5-20251001",
        anthropic_api_url="https://api.anthropic.com",
    ).to_json()

    delete_existing_evaluators(api_key)

    our_rule_ids = []
    for ev in EVALUATORS:
        rule_id = create_online_evaluator(api_key, ev, project_id, model_json)
        if rule_id:
            our_rule_ids.append(rule_id)

    print("\n  Every future trace will be automatically scored for:")
    for ev in EVALUATORS:
        print(f"    • {ev['feedback_key']}")

    return our_rule_ids


# ── Annotation queue ───────────────────────────────────────────────────────────

# Rubric shown to the reviewer in the annotation queue UI. `user_score` reuses
# the chat UI's 👍/👎 key so a vote cast in the queue lands on the same feedback
# key as a vote cast in the app; `factual_accuracy` mirrors the online evaluator
# of the same name, which is what demo step 4 is correcting by hand.
_QUEUE_RUBRIC = [
    {
        "feedback_key": USER_SCORE_KEY,
        "description": "Was this response helpful to the user?",
        "value_descriptions": {"1": "👍 Helpful", "0": "👎 Not helpful"},
        "is_required": True,
    },
    {
        "feedback_key": "factual_accuracy",
        "description": (
            "Are the LangChain/LangGraph/LangSmith facts correct? Flag stale doc "
            "domains (python.langchain.com, js.langchain.com) as inaccurate."
        ),
        "is_required": False,
    },
]

_QUEUE_INSTRUCTIONS = (
    "Review the agent's answer, correct it if it is wrong, then send the corrected "
    "example to the dataset so it becomes an offline eval case."
)

# How many recent root runs to scan for 👎 votes, and how many to enqueue.
_QUEUE_SCAN_LIMIT = 50
_QUEUE_SEED_LIMIT = 10


def _link_default_dataset(ls_client, queue_id) -> None:
    """Point the queue's 'add to dataset' action at the demo dataset.

    Demo step 4 sends corrected examples from the queue straight into the
    dataset; setting default_dataset makes that one click instead of a
    dataset picker. Only reachable over REST — create/update_annotation_queue
    don't expose the field — so it's best-effort and never fatal.
    """
    api_key = os.getenv("LANGSMITH_API_KEY", "")
    try:
        datasets = list(ls_client.list_datasets(dataset_name=DATASET_NAME))
        if not datasets:
            print(f"  Dataset '{DATASET_NAME}' not found — skipped default-dataset link.")
            return
        resp = requests.patch(
            f"https://api.smith.langchain.com/api/v1/annotation-queues/{queue_id}",
            headers=_ls_headers(api_key, json_body=True),
            json={"default_dataset": str(datasets[0].id)},
        )
        if resp.status_code in (200, 204):
            print(f"  Linked queue to dataset '{DATASET_NAME}'.")
        else:
            print(f"  Could not link default dataset ({resp.status_code}). Non-fatal.")
    except Exception as e:
        print(f"  Could not link default dataset: {e}. Non-fatal.")


def get_or_create_annotation_queue(ls_client):
    """Return the demo's annotation queue, creating it if it doesn't exist.

    Matched by exact name so re-running setup reuses the existing queue instead
    of stacking up duplicates in the shared workspace.
    """
    existing = next(
        (q for q in ls_client.list_annotation_queues(name=ANNOTATION_QUEUE_NAME)
         if q.name == ANNOTATION_QUEUE_NAME),
        None,
    )
    if existing:
        print(f"  Reusing existing queue '{ANNOTATION_QUEUE_NAME}'.")
        return existing

    queue = ls_client.create_annotation_queue(
        name=ANNOTATION_QUEUE_NAME,
        description=f"Human review of {PROJECT_NAME} traces for the Chat LangChain Lite demo.",
        rubric_instructions=_QUEUE_INSTRUCTIONS,
        rubric_items=_QUEUE_RUBRIC,
    )
    print(f"  Created queue '{ANNOTATION_QUEUE_NAME}'.")
    _link_default_dataset(ls_client, queue.id)
    return queue


def seed_annotation_queue(ls_client, queue_id, limit: int = _QUEUE_SEED_LIMIT) -> int:
    """Fill the queue with traces worth reviewing; returns how many were added.

    Prefers runs the user thumbed down in the chat UI — those are the ones the
    demo narrative is about. If nobody has voted yet (a fresh project), falls
    back to the most recent root runs so step 4 still has something to show.
    """
    runs = list(ls_client.list_runs(
        project_name=PROJECT_NAME, is_root=True, limit=_QUEUE_SCAN_LIMIT,
    ))
    if not runs:
        print("  No traces in the project yet — queue left empty.")
        return 0

    downvoted = {
        str(fb.run_id)
        for fb in ls_client.list_feedback(
            run_ids=[r.id for r in runs], feedback_key=[USER_SCORE_KEY],
        )
        if fb.score == 0
    }
    # Keep list_runs' newest-first ordering in both branches.
    picked = [r for r in runs if str(r.id) in downvoted][:limit]
    if picked:
        print(f"  Found {len(picked)} 👎-rated trace(s) to review.")
    else:
        picked = runs[:limit]
        print(f"  No 👎 votes yet — seeding {len(picked)} recent trace(s) instead.")

    # Already-queued runs are skipped so re-running setup doesn't double-add.
    queued = {str(r.id) for r in ls_client.list_runs_from_annotation_queue(queue_id)}
    to_add = [r.id for r in picked if str(r.id) not in queued]
    if to_add:
        ls_client.add_runs_to_annotation_queue(queue_id, run_ids=to_add)
    print(f"  Added {len(to_add)} run(s) to the queue ({len(picked) - len(to_add)} already queued).")
    return len(to_add)


def setup_annotation_queue() -> str:
    """Create the review queue and seed it. Returns the queue ID."""
    from langsmith import Client

    print(f"\n[4/5] Setting up annotation queue '{ANNOTATION_QUEUE_NAME}'...")
    ls_client = Client()
    queue = get_or_create_annotation_queue(ls_client)
    seed_annotation_queue(ls_client, queue.id)
    return str(queue.id)


# Context Hub plumbing lives in utils/context_hub.py — imported at call site.


# ── Baseline experiments ───────────────────────────────────────────────────────

# One baseline experiment per model. Both score ~100% on the permissive
# seed dataset; the demo beat is the cost/latency comparison between
# Haiku (cheap, fast) and Sonnet (more expensive, slower) in the
# Experiments view while the PR's CI is running.
_BASELINE_MODELS = [
    ("claude-haiku-4-5-20251001", "haiku"),
    ("claude-sonnet-4-6",         "sonnet"),
]


def seed_baseline_experiments() -> None:
    """Run one baseline experiment per model in _BASELINE_MODELS."""
    from scripts.run_evals import run_evaluation

    print(f"\n[5/5] Seeding {len(_BASELINE_MODELS)} baseline experiment(s) against '{DATASET_NAME}'...")

    for model_id, label in _BASELINE_MODELS:
        os.environ["CHAT_LANGCHAIN_LITE_MODEL"] = model_id
        prefix = f"baseline-{label}-chat-lc-lite-{DEMO_PRESENTER}"
        print(f"\n  → {model_id}: experiment '{prefix}-...'")
        scores = run_evaluation(experiment_prefix=prefix)
        print(f"  ✓ {label} complete: overall={scores.get('__overall__', 0.0):.2f}")

    os.environ.pop("CHAT_LANGCHAIN_LITE_MODEL", None)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-baseline-experiments",
        action="store_true",
        help="Skip seeding baseline experiments (faster setup, but the dataset's "
             "experiment list will be empty for the demo).",
    )
    args = parser.parse_args()

    api_key = os.getenv("LANGSMITH_API_KEY")
    if not api_key:
        print("Error: LANGSMITH_API_KEY not set.")
        sys.exit(1)

    from utils.context_hub import push_agents_md, push_demo_skills
    push_agents_md()
    push_demo_skills()
    ensure_project_exists()
    setup_dataset()
    our_rule_ids = setup_online_evaluators(api_key)
    queue_id = setup_annotation_queue()

    # Save state so cleanup can distinguish setup resources from Engine-added ones
    with open(".demo_state.json", "w") as f:
        json.dump({
            "run_rule_ids": our_rule_ids,
            "annotation_queue_id": queue_id,
        }, f, indent=2)

    if not args.skip_baseline_experiments:
        seed_baseline_experiments()

    print(f"\nSetup complete.")
    print(f"  Dataset:      {DATASET_NAME}")
    print(f"  Project:      {PROJECT_NAME}")
    print(f"  Online evals: scoring all new traces automatically")
    print(f"  Queue:        {ANNOTATION_QUEUE_NAME}")


if __name__ == "__main__":
    main()
