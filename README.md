# chat-lc-lite

A LangChain ecosystem chatbot ("Chat LangChain Lite") with intentional bugs, built to demonstrate LangSmith Engine's ability to identify issues in agent traces and propose fixes via PR. The agent answers questions about LangChain, LangGraph, LangSmith, and Deep Agents using three tools: `lookup_concept`, `get_setup_guide`, and `get_security_advice`.

## What this demos

1. **Engine identifies bugs** — the agent has bugs in the prompt and code that cause bad responses
2. **Engine proposes a PR fix** — targets the root cause code and opens a PR on your fork
3. **Engine proposes offline examples and online evals to add** — expand dataset coverage and monitoring with one click
4. **Offline evals in CI/CD** — the PR can't merge until eval scores pass a threshold
5. **Before/after scores in LangSmith** — both "before" and "after" experiments created automatically by CI when Engine opens a PR

## The bugs

Bugs are spread across three files so Engine has to reason about code, not just prompts:

| Bug | File / Location | Effect | Caught by |
|-----|------|--------|-----------|
| "Never use tools, never decline" instruction | LangSmith Context Hub (`chat-lc-lite-agent-robert` / AGENTS.md) — fix in the Context Hub UI, not the repo | Answers any topic; answers from memory instead of calling tools | `tool_usage`, `scope_adherence` |
| Casual / emoji voice | LangSmith Context Hub (`chat-lc-lite-agent-robert` / AGENTS.md) — fix in the Context Hub UI, not the repo | Every response starts with "Hey there! 👋", uses emojis throughout, ends with "Happy building! 🚀" | `professional_tone` |
| Wrong docs URL in SAFE_PATTERNS | `agent/tools.py` | Agent recommends stale `python.langchain.com` / `js.langchain.com` links instead of `docs.langchain.com` | `security_advice` |
| Wrong LangGraph min Python version | `agent/tools.py` | Returns "3.7+" instead of the correct "3.10+" | `factual_accuracy` |
| `max_tokens=300` | `agent/agent.py` | Truncates responses on complex technical questions | `response_completeness` |

## Setup

**1. Fork and clone this repo**

**2. Create a virtual environment**
```bash
uv sync
source .venv/bin/activate
```

Or with pip:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e . "langgraph-cli[inmem]"
```

> `uv sync` installs `langgraph-cli` (the `dev` dependency group) for you; the
> pip path installs it explicitly. The CLI runs the graph server that serves both
> the API and the chat UI (`langgraph dev`).

**3. Configure environment**
```bash
cp .env.example .env
```

Edit `.env`:
```
ANTHROPIC_API_KEY=your-key
LANGSMITH_API_KEY=your-demo-workspace-api-key
LANGSMITH_PROJECT=chat-lc-lite
LANGSMITH_WORKSPACE_ID=your-demo-workspace-id
LANGSMITH_TRACING=true
```

> If multiple presenters share a LangSmith workspace, use a unique `LANGSMITH_PROJECT` per person (e.g. `chat-lc-lite-morgan`) to avoid mixing traces and online evaluators. The project is created automatically on first use.

**4. Run one-shot setup**
```bash
python -m scripts.setup
```

This does four things in one command:
1. **Creates the LangSmith project** by sending one trace (required before online evaluators can be registered)
2. **Creates the dataset** `chat-lc-lite-scope-<your-name>` with 3 curated test cases, then tags that version as `baseline` in LangSmith
3. **Creates 6 online evaluators** in the LangSmith Evaluators UI at 100% sampling rate — every future trace is automatically scored for `security_advice`, `scope_adherence`, `tool_usage`, `response_completeness`, `professional_tone`, and `factual_accuracy`. Their run rule IDs are saved to `.demo_state.json` so cleanup can tell them apart from evaluators Engine adds.
4. **Creates the annotation queue** `chat-lc-lite-review-<your-name>` used in demo step 4, seeded with 👎-rated traces (or the 10 most recent, if nobody has voted yet). Its rubric covers `user_score` and `factual_accuracy`, and its default dataset is wired to the demo dataset so corrected examples are one click from becoming offline eval cases.

Only needs to be run once. Between demos, run `python -m scripts.cleanup` instead.

**5. Generate traces**
```bash
python -m scripts.generate_traces
```

Runs 11 single-turn queries and 1 multi-turn threaded conversation through the buggy agent to populate LangSmith with trace and thread variety beyond the dataset examples.

**6. Add GitHub secrets** (for CI/CD)

In your fork: Settings → Secrets → Actions → add `ANTHROPIC_API_KEY`, `LANGSMITH_API_KEY_GATEWAY`, `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT`, and `LANGSMITH_WORKSPACE_ID`.

> **Important:** When pasting secrets, make sure there are no trailing newlines or spaces.

**7. Enable GitHub Actions**

In your fork: Actions → (if prompted) enable workflows. GitHub disables Actions on forks by default — this step is required for offline evals to run on PRs.

**8. Connect Engine**

In LangSmith Engine, connect your LangSmith project (`LANGSMITH_PROJECT`) and your GitHub fork so Engine can read traces and open PRs against your repo.

## Demo flow

### Before the demo

```bash
# One-shot setup: creates dataset, online evaluators, annotation queue
python -m scripts.setup

# Generate more traces including threads
python -m scripts.generate_traces

# Start the chat UI (graph API + FastHTML frontend, served from one origin)
uv run langgraph dev
```

`uv run langgraph dev` boots the LangGraph server; the chat UI is mounted on it and
served at the server root (**http://localhost:2024/**). The graph API on the same
port powers streaming, thread history, and feedback.

### During the demo

1. Show Chat LangChain Lite UI — ask questions (concept lookups, setup guides, security advice, etc.); rate responses 👍/👎 to send feedback to LangSmith
2. Show traces in LangSmith with online eval scores (`security_advice`, `scope_adherence`, etc.)
3. Engine analyzes traces and identifies root causes across prompt and code
4. Add Engine-suggested offline examples — show ability to edit in the `chat-lc-lite-review-<your-name>` annotation queue, then send the corrected example to the dataset
5. Engine opens a PR on your fork
6. GitHub Actions runs evals on main (before experiment) and the PR branch (after experiment) — after scores pass ✅
7. Merge the PR
8. Add Engine-suggested online eval
9. Show the experiments in LangSmith — before/after score comparison

### After the demo

```bash
python -m scripts.cleanup
```

## The chat UI

The frontend (`web/app.py`) is a [FastHTML](https://fastht.ml) app mounted onto
the LangGraph server, so a single `langgraph dev` (or one deployment) serves both
the UI and the graph API from the same origin. It talks to the graph over the
LangGraph SDK on loopback — it never imports the graph directly.

- **Streaming chat** — responses stream token-by-token over SSE, rendered as
  markdown client-side (sanitized with DOMPurify).
- **Feedback** — rate any response 👍/👎 and leave an optional comment; feedback
  is written to LangSmith (as `user_score` / `user_comment`) keyed to the run, so
  it shows up on the trace and feeds online evals.
- **Trace deep-link** — every response has a ↗ Trace link that opens its LangSmith
  trace.
- **Thread history** — a sidebar lists prior conversations; reopening a thread
  rebuilds it from the graph's persisted state, with each response's stored vote
  restored.

## Scripts

| Script | What it does |
|--------|-------------|
| `python -m scripts.setup` | One-shot setup: creates dataset, 6 online evaluators, and the annotation queue |
| `python -m scripts.generate_traces` | Runs 11 single-turn queries + 1 multi-turn thread through the buggy agent |
| `python -m scripts.run_evals` | Runs offline evals against the dataset and prints scores |
| `python -m scripts.run_evals --skip-dataset` | Re-runs evals against existing dataset (used in CI) |
| `python -m scripts.run_evals --threshold 0.7` | Exits with code 1 if scores < 0.7 (used in CI) |
| `python -m scripts.cleanup` | Resets demo to clean state — see Cleanup section |
| `python -m scripts.cleanup --full` | Same, plus deletes the LangSmith project and annotation queue (so Engine sees a fresh project on the next demo). Re-run `scripts.setup` after. |
| `uv run langgraph dev` | Start the graph server with the Chat LangChain Lite UI mounted on it (http://localhost:2024/) |

## Evaluators

Two LLM-as-judge evaluators run in CI (offline). Claude Haiku scores each 0 or 1:

- **`tool_selection`** — did the agent ground its response in tool output rather than answering from memory? Goes 0→1 when the bad system prompt is fixed.
- **`scope_adherence`** — did the agent stay LangChain-ecosystem-only and decline off-topic questions?

## Online Evaluators

Online evaluators run automatically on every trace as it arrives in LangSmith. This gives Engine a continuous signal on live traffic, not just offline evals on a fixed dataset.

Six online evaluators are registered by `python -m scripts.setup`: `security_advice`, `scope_adherence`, `tool_usage`, `response_completeness`, `professional_tone`, and `factual_accuracy`.

## CI/CD

`.github/workflows/evals.yml` runs automatically on every PR to `main`.

Add these secrets to your repo (Settings → Secrets → Actions):
- `ANTHROPIC_API_KEY` — the gateway key (`lsv2_sk_…`) the LLM-as-judge evaluator uses
- `LANGSMITH_API_KEY_GATEWAY` — the gateway key `utils/models.py` routes the agent through
- `LANGSMITH_API_KEY`
- `LANGSMITH_PROJECT`
- `LANGSMITH_WORKSPACE_ID`

`LANGSMITH_PROJECT` should match what you used locally — that's the project the agent traces against.

```
PR opened → GitHub Actions → run_evals --skip-dataset --threshold 0.7
                                          ↓
                               scores < 0.7 → ❌ blocks merge
                               scores ≥ 0.7 → ✅ mergeable
```

CI runs evals on both the base branch (creating the "before" experiment) and the PR branch (creating the "after" experiment) in LangSmith automatically. Because `--skip-dataset` fetches the existing dataset from LangSmith by name, any examples Engine adds to the dataset are included in the eval run automatically.

## Repo structure

```
context/
└── __init__.py       # get_prompt() — pulls the agent's system prompt from
                      # LangSmith Context Hub at runtime. The prompt content
                      # lives in the hub, not this repo (Bugs 1 & 5 are fixed
                      # in the Context Hub UI, not via code PR).

agent/
├── tools.py          # concept lookup, setup guides, security advice (Bugs 2 & 3)
└── agent.py          # create_agent + FilesystemMiddleware (Bug 4 — max_tokens)

utils/
└── context_hub.py    # setup-time push helper. Holds the *initial seed* for
                      # Context Hub only; not the runtime source of truth.

evals/
├── dataset.py        # creates per-user LangSmith dataset (3 curated examples)
└── evaluators.py     # 2 LLM-as-judge offline evaluators (used in CI)

scripts/
├── setup.py          # one-shot setup: dataset + online evaluators + annotation queue + Context Hub
├── generate_traces.py    # populate LangSmith with extra traces and threads
├── run_evals.py          # offline evals + CI threshold check
└── cleanup.py            # resets demo to clean state after presentation

.github/workflows/
└── evals.yml                 # CI/CD: offline evals on PRs to main, gated on the
                              # manually-applied 'run-evals' label

web/
└── app.py           # Chat LangChain Lite UI (FastHTML). Mounted onto the graph
                     # server via langgraph.json's `http.app`, so the UI and the
                     # graph API are served from the same origin.

langgraph.json       # LangGraph deployment manifest: exposes the `agent` graph
                     # and mounts the FastHTML UI (`http.app`).
```

## Cleanup

Run after the demo to reset everything for the next presenter:

```bash
python -m scripts.cleanup
```

This does six things:
1. **Resets dataset to original 3 examples** — deletes all examples and re-uploads the canonical 3, removing anything Engine added
2. **Deletes CI/Engine experiments** — keeps the `baseline-*` seed experiments from `setup.py` (the Haiku-vs-Sonnet "before" reference); CI/CD regenerates before/after experiments on every PR
3. **Removes Engine-added online evaluators** — uses saved run rule IDs from `.demo_state.json` to delete only evaluators Engine added, leaving the 5 from `setup.py` in place
4. **Resets the annotation queue** — empties it and re-seeds from 👎-rated traces, since reviewing an item during the demo marks it done and drains the queue. Feedback left on the traces themselves is kept; only queue membership is reset.
5. **Re-seeds Context Hub to the buggy baseline** — re-pushes the seed `AGENTS.md` and demo skills, restoring the buggy prompt if it was fixed in the Context Hub UI during the demo (a code/dataset reset can't touch Context Hub)
6. **Resets main to the `baseline` tag** — force-resets to remove Engine's merged PR, restoring the buggy agent state

After cleanup, the demo is ready to run again — no need to re-run `setup.py`.

For a **full** reset that also removes the LangSmith project and annotation queue (clearing all traces and Engine's per-project issue state):

```bash
python -m scripts.cleanup --full
python -m scripts.setup         # recreates project, dataset, evaluators, queue
python -m scripts.generate_traces
```

Use this when you want Engine to see a completely fresh project for the next demo — for example when a new presenter takes over and you don't want them to inherit any pre-flagged issues.
