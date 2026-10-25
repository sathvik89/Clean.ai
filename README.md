# Clean.ai

An autonomous, human-supervised data cleaning agent. Upload a messy CSV, tell it
what the data is for, review the plan it proposes, and get back a clean dataset —
with proof of every change and a reusable script to do it again without the app.

Built on LangGraph + Groq, with a Streamlit front end.

## What it does

1. **Analyze** — deterministic profiling builds a *column card* for every column
   (role, type, key-ness, sample values) and detects real problems: missing
   values, disguised nulls (`"N/A"`, `"-"`, `"null"` hiding as text), duplicate
   rows and keys, wrong dtypes, invalid emails/dates, impossible ages, negative
   salaries, future dates, mixed types.
2. **Plan** — the LLM produces a structured JSON plan (tool, columns, params,
   rationale, priority per step), validated against the tool registry. Unusable
   plans fall back to a deterministic plan, so the run never dies on bad JSON.
3. **Human review** — the run pauses. You see the issues table, the full plan in
   an editable grid, and a **dry-run impact preview** ("this step would drop 3
   rows, fix 24 nulls") before anything touches your data. Edit cells, untick
   steps, or just type "don't drop any rows" — the plan is revised in place.
4. **Execute** — the agent works step by step through the plan with tool calls,
   streaming a live timeline. Every action is recorded in a ledger with its
   measured effect.
5. **Validate & repair** — a deterministic validator re-profiles the result; if
   issues remain, the agent enters a repair loop focused on exactly those issues.
6. **Deliver** — quality scores before/after (overall + four dimensions),
   comparison charts, the cleaned CSV, a markdown run report, the plan as JSON,
   and a **recipe**: a standalone pandas script that reproduces the cleaning on
   next month's file with no API key.

## Engineering highlights

**Context engineering.** The executor never sees raw chat history. Each turn is
rebuilt from structured state: current plan step, column cards (semantic
memory), a compact action ledger (episodic memory), the last tool result
(clipped), and validation feedback. This keeps prompts small, deterministic,
and cache-friendly.

**Layered memory.**
- *Column cards* — per-column knowledge (role, open issues, fixes applied),
  refreshed deterministically after every mutation.
- *Action ledger* — one compact line per tool call with the measured delta
  (`nulls.age 24->0`), rendered with old entries collapsed.
- *Case memory* — finished runs store a dataset fingerprint (roles + issue
  kinds) and which strategies worked, in SQLite. Future runs on similar-looking
  data retrieve those as planner hints via weighted Jaccard matching — including
  negative hints ("run_pandas FAILED 5/8 times, avoid it"). No embeddings, fully
  deterministic and debuggable.

**Safety and guardrails.**
- LLM-written pandas snippets run through an **AST sandbox**: no imports, no
  dunders, no file/network/system access, restricted builtins.
- A **row-deletion cap**: any single tool call that would delete more than a
  configurable share of rows is refused with an explanation the model can act on.
- **Duplicate-call detection**: an identical tool call on identical data is
  short-circuited with a nudge to try something different.
- **Stagnation detection** plus hard caps on tool and validation rounds, and
  per-step attempt limits with an explicit `report_blocked` escape hatch.
- Sample values shown to the LLM are **PII-masked** (`r***@corp.com`).

**Single tool registry.** Each tool's name, description, pydantic argument
schema, and handler are declared once. The same registry builds the LLM tool
bindings, validates calls, dispatches execution, and powers the dry-run
simulator — the schema the model sees can never drift from the code that runs.

**Observability.** Token usage is tracked per run and stored with the full run
record (plan, ledger, logs, analysis, cleaned data) in SQLite, per user.

## Project layout

```
src/cleanai/
  config.py             settings, limits, model config
  core/
    profiling.py        column cards, issue detection, quality scores, snapshots
    memory.py           action ledger, card rendering, SQLite case memory
    plan.py             structured plan parsing and progress tracking
  agents/
    prompts.py          versioned prompt set
    state.py            graph state
    llm.py              Groq factory + retry
    nodes.py            analyzer / planner / executor / tools / validator
    graph.py            LangGraph wiring with human-review interrupt
    tools/
      registry.py       single source of truth for all agent tools
      cleaning.py       the pandas transformations
      sandbox.py        AST-guarded snippet execution
  services/
    pipeline.py         CleaningSession orchestration, dry-run, chat plan edits
    persistence.py      per-user run history (SQLite)
    auth.py             bcrypt accounts (SQLite)
    recipe.py           standalone-script + plan JSON export
    reporting.py        markdown run reports
app/                    Streamlit UI (login, workspace, history, run detail, insights)
tests/                  unit tests + scripted-LLM full-graph integration tests
scripts/smoke_test.py   live end-to-end test against the Groq API
data/samples/           three deliberately messy demo datasets
```

## Getting started

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env        # add your GROQ_API_KEY

streamlit run app/streamlit_app.py            # web app
python main.py data/samples/hr_employees.csv --goal "ML modeling"   # CLI
```

## Testing

```bash
pytest                          # 62 tests, no API key needed
python scripts/smoke_test.py    # live end-to-end run against Groq
```

The integration tests drive the entire graph with a scripted fake LLM, so the
agent loop — plan tracking, guardrails, duplicate detection, repair routing —
is verified deterministically on every run.
