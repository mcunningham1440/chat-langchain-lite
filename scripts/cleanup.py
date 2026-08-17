"""Clean up LangSmith resources after the Chat LangChain Lite demo.

Resets the demo to a clean state so it can be run again without re-running setup:
  1. Resets dataset to the original 3 examples (deletes Engine-added examples)
  2. Deletes CI/Engine experiments (keeps the baseline-* seeds from setup.py)
  3. Removes Engine-added online evaluators (keeps the 5 registered by setup.py)
  4. Empties and re-seeds the annotation queue (clears any reviews done during
     the demo, so step 4 has unreviewed traces again)
  5. Re-seeds Context Hub (AGENTS.md + demo skills) to the buggy baseline,
     restoring the prompt if it was fixed in the Context Hub UI during the demo
  6. Force-resets main back to the 'baseline' tag (removes Engine's merged PR)

Optional: --full also deletes the LangSmith project and annotation queue
entirely (clears all traces and Engine's per-project issue state). After a full reset, re-run
`python -m scripts.setup` before the next demo to recreate the project,
dataset, and online evaluators from scratch.

Usage:
    python -m scripts.cleanup          # standard reset
    python -m scripts.cleanup --full   # also delete the LangSmith project
"""

import argparse
import json
import os
import subprocess
import sys
import time
import requests
from dotenv import load_dotenv

load_dotenv(override=True)

from evals.dataset import DATASET_NAME, TOOL_ADHERENCE_DATASET_NAME, DEMO_PRESENTER
PROJECT_NAME = os.getenv("LANGSMITH_PROJECT", "chat-lc-lite")


# ── 1. Reset dataset ───────────────────────────────────────────────────────────

def reset_dataset() -> None:
    """Reset both demo datasets to their canonical seed examples.

    Engine may add examples to the primary dataset, so we delete everything
    and re-upload the originals.
    """
    from langsmith import Client
    from evals.dataset import EXAMPLES, TOOL_ADHERENCE_EXAMPLES

    print(f"\n[1/4] Resetting demo datasets to canonical seeds...")
    ls_client = Client()

    for name, examples in (
        (DATASET_NAME, EXAMPLES),
        (TOOL_ADHERENCE_DATASET_NAME, TOOL_ADHERENCE_EXAMPLES),
    ):
        datasets = list(ls_client.list_datasets(dataset_name=name))
        if not datasets:
            print(f"  Dataset '{name}' not found. Skipping.")
            continue
        dataset = datasets[0]
        existing = list(ls_client.list_examples(dataset_id=dataset.id))
        if existing:
            ls_client.delete_examples([e.id for e in existing])
        ls_client.create_examples(
            dataset_id=dataset.id,
            inputs=[e["input"] for e in examples],
            outputs=[e["output"] for e in examples],
            metadata=[e.get("metadata", {}) for e in examples],
        )
        print(f"  '{name}': cleared {len(existing)}, re-uploaded {len(examples)}.")


# ── 2. Delete 'after' experiments ─────────────────────────────────────────────

def delete_ci_experiments() -> None:
    """Delete CI/Engine experiments linked to the dataset.

    CI/CD generates fresh before/after experiments on every PR, so those are
    not worth keeping between demos. The `baseline-*` experiments seeded by
    setup.py ARE preserved: they're the demo's Haiku-vs-Sonnet "before"
    reference and would otherwise have to be reseeded after every cleanup.
    """
    from langsmith import Client

    print(f"\n[2/4] Removing CI/Engine experiments (keeping baseline seeds)...")
    ls_client = Client()
    total_deleted = 0
    total_kept = 0
    for name in (DATASET_NAME, TOOL_ADHERENCE_DATASET_NAME):
        datasets = list(ls_client.list_datasets(dataset_name=name))
        if not datasets:
            continue
        experiments = list(ls_client.list_projects(reference_dataset_id=datasets[0].id))
        for exp in experiments:
            # Preserve the baseline-* seeds setup.py creates (the Haiku/Sonnet
            # "before" reference); only sweep CI/Engine experiments.
            if exp.name.startswith("baseline-"):
                total_kept += 1
                continue
            for attempt in range(3):
                try:
                    ls_client.delete_project(project_name=exp.name)
                    total_deleted += 1
                    time.sleep(1.0)
                    break
                except Exception as e:
                    if "429" in str(e) and attempt < 2:
                        time.sleep(5.0)
                    else:
                        print(f"  Failed to delete '{exp.name}': {e}")
                        break
    print(f"  Deleted {total_deleted} experiment(s), kept {total_kept} baseline seed(s).")


# ── 3. Delete Engine-added online evaluators ───────────────────────────────────

def delete_engine_evaluators(api_key: str) -> None:
    """Delete only the online evaluators Engine added — leave our setup.py ones intact.

    setup.py saves the run rule IDs it created to .demo_state.json.
    Any run rule scoped to our project that is NOT in that list was added by
    Engine and is safe to remove. Rules belonging to other projects are ignored.
    """
    from langsmith import Client

    print(f"\n[3/4] Removing Engine-added online evaluators...")

    try:
        with open(".demo_state.json") as f:
            state = json.load(f)
        our_rule_ids = set(state.get("run_rule_ids", []))
    except FileNotFoundError:
        print("  Warning: .demo_state.json not found — run 'python -m scripts.setup' first.")
        print("  Skipping online evaluator cleanup.")
        return

    # Get our project ID so we only touch rules scoped to our project
    # Filter server-side: an unfiltered list_projects() paginates the whole
    # workspace, which in a shared workspace is thousands of projects and reads
    # until the socket times out. name= narrows it to one page.
    ls_client = Client()
    project = next(
        (p for p in ls_client.list_projects(name=PROJECT_NAME) if p.name == PROJECT_NAME),
        None,
    )
    if not project:
        print(f"  Warning: project '{PROJECT_NAME}' not found. Skipping.")
        return
    project_id = str(project.id)

    resp = requests.get(
        "https://api.smith.langchain.com/api/v1/runs/rules",
        headers={"x-api-key": api_key},
    )
    if resp.status_code != 200:
        print(f"  Could not list run rules ({resp.status_code}). Skipping.")
        return

    deleted = 0
    for rule in resp.json():
        # Only consider rules tied to our project
        if rule.get("session_id") != project_id:
            continue
        if rule["id"] not in our_rule_ids:
            r = requests.delete(
                f"https://api.smith.langchain.com/api/v1/runs/rules/{rule['id']}",
                headers={"x-api-key": api_key},
            )
            if r.status_code in (200, 204):
                print(f"  Deleted Engine run rule '{rule.get('display_name', rule['id'])}'")
                deleted += 1

    if deleted == 0:
        print("  No Engine-added evaluators found.")
    else:
        print(f"  Deleted {deleted} Engine-added evaluator(s).")


# ── 4. Reset the annotation queue ──────────────────────────────────────────────

def reset_annotation_queue() -> None:
    """Empty the demo's annotation queue and re-seed it with traces to review.

    Reviewing a run in the queue marks it done and removes it from the pending
    list, so after a demo the queue is drained. Like reset_dataset(), this
    clears whatever is left and re-seeds from the same source setup.py uses.
    Feedback left on the reviewed runs stays on the traces — only the queue
    membership is reset.
    """
    from langsmith import Client
    from scripts.setup import (
        ANNOTATION_QUEUE_NAME, get_or_create_annotation_queue, seed_annotation_queue,
    )

    print(f"\n[4/4] Resetting annotation queue '{ANNOTATION_QUEUE_NAME}'...")
    ls_client = Client()
    try:
        queue = get_or_create_annotation_queue(ls_client)
        stale = list(ls_client.list_runs_from_annotation_queue(queue.id))
        for run in stale:
            ls_client.delete_run_from_annotation_queue(queue.id, run_id=run.id)
        if stale:
            print(f"  Cleared {len(stale)} run(s) from the queue.")
        seed_annotation_queue(ls_client, queue.id)
    except Exception as e:
        print(f"  Annotation queue reset failed (non-fatal): {e}")


# ── Optional: delete the entire LangSmith project ─────────────────────────────

def delete_project() -> None:
    """Delete this presenter's project + datasets + Context Hub repo.

    Removes all traces, online evaluators on the project, Engine issue state,
    both demo datasets, and the presenter's Context Hub agent repo.

    Datasets and Context Hub repos are matched by the `-<presenter>` suffix
    so running `--full` in a shared LangSmith workspace never deletes another
    demoer's resources.
    """
    from langsmith import Client

    ls_client = Client()

    # 1. Project
    print(f"\n[*] Deleting LangSmith project '{PROJECT_NAME}'...")
    try:
        ls_client.delete_project(project_name=PROJECT_NAME)
        print(f"  Deleted project '{PROJECT_NAME}'.")
    except Exception as e:
        if any(s in str(e).lower() for s in ("not found", "404")):
            print(f"  Project '{PROJECT_NAME}' not found.")
        else:
            print(f"  Project delete failed: {e}")

    # 2. Datasets — delete entirely (not reset). Scope every match to THIS
    # presenter so a shared workspace's other demoers keep their datasets:
    #   - the two current demo datasets (exact names), plus
    #   - any stale chat-lc-lite-*-<presenter> datasets from prior renames
    #     (the `-<presenter>` suffix is what setup.py appends), plus
    #   - the auto-generated `Evaluator: <PROJECT_NAME>:...` pseudo-datasets
    #     LangSmith creates when this project's online evaluators run (they
    #     linger after the evaluator itself is deleted).
    # The bare `chat-lc-lite-` prefix is deliberately NOT used: it ignores the
    # presenter suffix and would delete other demoers' datasets.
    print(f"\n[*] Deleting demo datasets...")
    known = {DATASET_NAME, TOOL_ADHERENCE_DATASET_NAME}
    presenter_suffix = f"-{DEMO_PRESENTER}"
    for d in ls_client.list_datasets():
        mine = (
            d.name in known
            or (d.name.startswith("chat-lc-lite-") and d.name.endswith(presenter_suffix))
            or d.name.startswith(f"Evaluator: {PROJECT_NAME}")
        )
        if not mine:
            continue
        try:
            ls_client.delete_dataset(dataset_id=d.id)
            print(f"  Deleted dataset '{d.name}'.")
        except Exception as e:
            print(f"  Dataset delete failed for '{d.name}': {e}")

    # 3. Context Hub — delete only THIS presenter's repos. The agent repo is
    # `chat-lc-lite-agent-<presenter>`, so the same `chat-lc-lite-*` +
    # `-<presenter>` suffix match used for datasets above scopes this safely.
    # The bare `chat-lc-lite-` prefix and the hardcoded generic skill handles
    # (release-notes-skill, etc.) are deliberately NOT used: those handles are
    # seeded un-scoped in utils/context_hub.py and would delete other demoers'
    # identically-named repos in a shared workspace.
    print(f"\n[*] Deleting Context Hub repos (presenter '{DEMO_PRESENTER}')...")
    api_key = os.environ.get("LANGSMITH_API_KEY", "")
    workspace_id = os.environ.get("LANGSMITH_WORKSPACE_ID", "")
    H = {"x-api-key": api_key}
    if workspace_id:
        H["X-Tenant-Id"] = workspace_id
    for repo_type, delete_fn in (("agent", ls_client.delete_agent), ("skill", ls_client.delete_skill)):
        r = requests.get(
            f"https://api.smith.langchain.com/v1/platform/hub/repos?repo_type={repo_type}",
            headers=H,
        )
        if r.status_code != 200:
            continue
        for repo in r.json().get("repos", []):
            handle = repo.get("repo_handle", "")
            if handle.startswith("chat-lc-lite-") and handle.endswith(presenter_suffix):
                try:
                    delete_fn(handle)
                    print(f"  Deleted {repo_type} '{handle}'.")
                except Exception as e:
                    if not any(s in str(e).lower() for s in ("not found", "404")):
                        print(f"  {repo_type} delete failed for '{handle}': {e}")

    # 4. Annotation queues — workspace-level, so deleting the project leaves
    # them behind. Scoped to this presenter with the same `chat-lc-lite-*` +
    # `-<presenter>` match used for datasets and Hub repos above.
    print(f"\n[*] Deleting annotation queues (presenter '{DEMO_PRESENTER}')...")
    from scripts.setup import ANNOTATION_QUEUE_NAME
    for q in ls_client.list_annotation_queues():
        mine = q.name == ANNOTATION_QUEUE_NAME or (
            q.name.startswith("chat-lc-lite-") and q.name.endswith(presenter_suffix)
        )
        if not mine:
            continue
        try:
            ls_client.delete_annotation_queue(q.id)
            print(f"  Deleted annotation queue '{q.name}'.")
        except Exception as e:
            print(f"  Annotation queue delete failed for '{q.name}': {e}")

    # Clear the saved run-rule / queue IDs from state — those IDs no longer exist
    try:
        os.remove(".demo_state.json")
        print("  Removed stale .demo_state.json.")
    except FileNotFoundError:
        pass


# ── 4. Reset repo main to the baseline tag ────────────────────────────────────

def reset_main_to_baseline() -> None:
    """Force-reset the repo's main branch to the `baseline` tag.

    The `baseline` tag is pinned to the initial buggy commit when the repo is
    created. After Engine opens (and you merge) a PR, this resets main to that
    pinned commit so the demo can be re-run from scratch. Unlike the previous
    fork-vs-upstream design, the tag lives in the same repo — no external
    upstream needed.
    """
    # Not part of the numbered sequence: this runs in both the standard and
    # --full paths, so a fixed step number would be wrong in one of them.
    print(f"\n[*] Resetting main branch to the 'baseline' tag...")

    # Get the repo from origin URL
    result = subprocess.run(
        ["git", "remote", "get-url", "origin"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print("  Warning: could not determine origin remote. Skipping.")
        return

    fork_url = result.stdout.strip()
    if "github.com:" in fork_url:
        fork_repo = fork_url.split("github.com:")[1].removesuffix(".git")
    elif "github.com/" in fork_url:
        fork_repo = fork_url.split("github.com/")[1].removesuffix(".git")
    else:
        print(f"  Warning: unrecognised remote URL '{fork_url}'. Skipping.")
        return

    # Get the baseline tag's commit SHA from GitHub
    result = subprocess.run(
        ["gh", "api", f"repos/{fork_repo}/git/refs/tags/baseline", "--jq", ".object.sha"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"  Warning: 'baseline' tag not found on '{fork_repo}' ({result.stderr.strip()}). Skipping.")
        return
    baseline_sha = result.stdout.strip()

    # Tag points to a tag object for annotated tags — dereference to commit
    if baseline_sha:
        deref = subprocess.run(
            ["gh", "api", f"repos/{fork_repo}/git/tags/{baseline_sha}", "--jq", ".object.sha"],
            capture_output=True, text=True,
        )
        if deref.returncode == 0 and deref.stdout.strip():
            baseline_sha = deref.stdout.strip()

    # Force-reset main to baseline SHA
    result = subprocess.run(
        ["gh", "api", f"repos/{fork_repo}/git/refs/heads/main",
         "-X", "PATCH",
         "-f", f"sha={baseline_sha}",
         "-F", "force=true"],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        print(f"  Reset '{fork_repo}' main to baseline ({baseline_sha[:8]}).")
    else:
        print(f"  Warning: reset failed ({result.stderr.strip()}).")


# ── Re-seed Context Hub ───────────────────────────────────────────────────────

def reset_context_hub() -> None:
    """Re-seed Context Hub (AGENTS.md + demo skills) back to the buggy baseline.

    Standard cleanup resets code, dataset, and experiments — but the agent's
    system prompt lives in Context Hub and is edited via the UI during the demo.
    If a presenter applied the prompt fix live, this restores the buggy seed so
    the next demo starts clean. Re-pushing overwrites any edited AGENTS.md; the
    seed content is owned by utils/context_hub.py (the same source setup.py uses).

    Not called by --full, which deletes the Hub repos for a clean-slate re-setup.
    """
    from utils.context_hub import push_agents_md, push_demo_skills

    print(f"\n[*] Resetting Context Hub to seed (AGENTS.md + demo skills)...")
    try:
        push_agents_md()
        push_demo_skills()
    except Exception as e:
        print(f"  Context Hub reset failed (non-fatal): {e}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--full",
        action="store_true",
        help="Also delete the LangSmith project (clears all traces and Engine's "
             "per-project issue state). Re-run scripts.setup after.",
    )
    args = parser.parse_args()

    api_key = os.getenv("LANGSMITH_API_KEY")
    if not api_key:
        print("Error: LANGSMITH_API_KEY not set.")
        sys.exit(1)

    print(f"Cleaning up demo...")
    print(f"  Dataset:  {DATASET_NAME}")
    print(f"  Project:  {PROJECT_NAME}")
    if args.full:
        print(f"  Mode:     FULL (project will be deleted)")

    if args.full:
        # --full: nuke everything. Experiments are projects too, so delete
        # them first; then the main tracing project, datasets, and Context
        # Hub repos.
        delete_ci_experiments()
        delete_project()
    else:
        reset_dataset()
        delete_ci_experiments()
        delete_engine_evaluators(api_key)
        reset_annotation_queue()
        reset_context_hub()
    reset_main_to_baseline()

    if args.full:
        print(f"\nFull cleanup complete. Run `python -m scripts.setup` before the next demo.")
    else:
        print(f"\nCleanup complete. Ready for the next demo.")


if __name__ == "__main__":
    main()
