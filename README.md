# Clean.ai

An autonomous, human-supervised data cleaning agent. Upload a messy CSV, tell it
what the data is for, review the plan it proposes, and get back a clean dataset —
with proof of every change and a reusable script to do it again without the app.

Built on LangGraph + Groq, with a Streamlit front end.

## What it does

1. **Explore (agent-led EDA)** — deterministic profiling builds a *column card*
   for every column and a full EDA bundle (distributions, correlations,
   missingness, category frequencies). The agent then *investigates* anything
   surprising with read-only inspection tools it chooses itself — `value_counts`,
   `describe`, `correlation_pairs`, `sample_rows` — before writing a narrative
   EDA report backed by charts.
2. **Suggest** — from the EDA, the agent proposes cleaning **suggestions**: each
   one a card with a title, reason, severity, and the exact tool + parameters it
   maps to. Detected issues cover missing values, disguised nulls (`"N/A"`,
   `"-"`), duplicates, wrong dtypes, invalid emails/dates, impossible values and
   more. If the model's JSON is unusable, deterministic fallback suggestions
   keep the run alive.
3. **You decide** — untick anything you disagree with, see a **dry-run impact
   preview** ("would drop 3 rows, fix 24 nulls") before anything touches your
   data, and add your own asks in plain English ("also lowercase emails, never
   drop rows") — they become first-class suggestions too. Accepted suggestions
   compile deterministically into an ordered plan: structure fixes before value
   fixes before cosmetics.
4. **Execute** — the agent works the plan step by step with its 23-tool kit,
   streaming a live timeline. Every action lands in a ledger with its measured
   effect.
5. **Validate & repair** — a deterministic validator re-profiles the result; if
   issues remain, the agent enters a repair loop focused on exactly those issues.
6. **Deliver** — quality scores before/after (overall + four dimensions),
   comparison charts, a per-column scorecard, highlighted diffs, the cleaned
   CSV, a markdown report, the plan as JSON, and a **recipe**: a standalone
   pandas script that reproduces the cleaning on next month's file with no API
   key. Past recipes can be re-applied to new files instantly from run history.

## Measured performance

Numbers from live runs against Groq `openai/gpt-oss-120b` (see
`scripts/smoke_test.py` and `scripts/scale_test.py`):

| benchmark | result |
|---|---|
| 153,000-row messy dataset | quality **22 → 100**, 38,373 missing cells fixed, status *clean* |
| wall time for that run | **102 s** end to end |
| LLM cost at 153k rows | **~26k tokens** — same order as an 8-row file, because the model only ever sees compact profiles, never the data |
| 8-row seeded-chaos dataset | quality **15 → 100**, status *clean* |
| automated tests | **75 passing**, including scripted-LLM full-graph integration tests |
| agent toolkit | **23 tools** (17 cleaning + 6 read-only inspection) |

Dataset size is bounded by your machine's RAM, not by the model: profiling
samples 50k rows for the expensive statistics while keeping missing/duplicate
counts exact.

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
The kit goes well beyond drop/fill: group-wise imputation, category
standardization (fuzzy label merging), number extraction from text like
'₹55,000', robust date parsing, explicit value mapping, and flag-instead-of-drop
for suspicious rows.

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
