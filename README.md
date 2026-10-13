# GenAI Test Case & Synthetic Test Data Generation (Langflow + Claude)

Turns **release notes and requirement specs** into **requirement-aligned, traceable test cases**.
An **independent judge model** checks which acceptance criteria each case really tests.
Coverage gaps are closed automatically, and cases go to **JIRA for human review**.
Finally, it generates **realistic synthetic test data** that feeds straight into the automated test runner.

It is built to run on **every release**:
- Only new or changed requirements are sent to the LLM.
- Requirement and test-case IDs stay stable between runs.
- JIRA issues are updated, never duplicated.
- An approval only counts for the content the reviewer actually approved.
- Edits a reviewer makes in JIRA are pulled back into the suite.

The same `testgen` package runs in two places:
- **Langflow**, as four custom nodes, for prototyping and demos.
- **CLI / CI**, via `python -m testgen.pipeline`, for production.

---

## 1. Architecture

```
 Release notes ─┐                                            state/state.json
 Req. specs ────┼─► [1] Requirement Extractor ─► ID registry ─► new / changed / unchanged / removed
                ┘     (previous reqs in prompt     (module+title fuzzy match;     │
                       for wording stability)       any changed number = changed) │
                                                                                  ▼
 CSV / JSON ─► consolidate (join on *_id) ─► schema profile      only NEW + CHANGED requirements:
                                                   │            ┌──────────────────────────────────┐
                                                   └──────────► │ [2] generate (per requirement)   │
                                                                │ [3] dedupe (same data+outcome)   │
                                                                │ [4] JUDGE verifies each case     │
                                                                │     → verified criteria          │
                                                                │     → grounded? accept/revise/   │
                                                                │       reject                     │
                                                                │ [5] gaps on VERIFIED coverage    │
                                                                │     → re-prompt only for gaps    │
                                                                └───────────────┬──────────────────┘
                                                                                ▼
                     stable test-case IDs (regenerated cases inherit the ID of the matching old case)
                                                                                ▼
 [6] JIRA: pull reviews first (reviewer edits win) → create / update / skip / obsolete
           approval is voided when the content changes; rejected cases never run
                                                                                ▼
 [7] Synthetic data: joined reference records filtered by state, cross-field rules
     (amount ≤ balance, destination == source), setup data (prior transfers with exact totals)
                                                                                ▼
     output/testdata/<TC-ID>.json + manifest.json ──► pytest / REST Assured / Playwright
     output/eval_metrics.json, traceability.csv, verification.json, audit_log.jsonl
```

---

## 2. What makes the results trustworthy

| Problem | How it is handled | Where |
|---|---|---|
| The generator marks its own homework (`covers_criteria`) | A separate judge prompt (`SYSTEM_REVIEWER`) re-reads every case and returns `verified_criteria`. **Coverage, the gap loop and the metrics use only judge-verified criteria.** Criteria a case claims but the judge rejects are reported as `claimed_but_unverified`. | `verify.py`, `coverage.py` |
| Invented expectations | The judge marks a case ungrounded if any value or message isn't in the requirement. Ungrounded cases are rejected, and `revise` cases go to JIRA labelled `needs-revision` with the reason. | `verify.py`, `jira_publisher.py` |
| "Are the tests any good?" | `evals/golden.json` lists must-have scenarios (e.g. `amount=10000.01 → rejection`). The metrics are: verified vs claimed coverage, over-claim rate, grounded rate, duplicate rate, judge-reject rate, reviewer-rejection rate, and golden recall. | `evaluate.py` |
| Prompt changes silently getting worse | `python -m testgen.evaluate` compares a run with `evals/baseline.json` and **exits 1 on regression**, so it can gate CI. | `evaluate.py` |
| Boundary semantics | "up to N" is a maximum: N is accepted and N+0.01 is rejected. "above N require OTP" is a threshold: N+0.01 triggers a *challenge*, not a rejection. Every limit is tested on both sides. | `offline.py`, prompts |
| Dedupe dropping real boundary cases | Two cases count as duplicates only if they use the **same data and expect the same outcome**. 1.00 and 1.01 are never merged. | `coverage.dedupe` |

## 3. What makes it safe to run on every release

| Problem | How it is handled | Where |
|---|---|---|
| IDs reshuffle each run | Requirement IDs come from a registry keyed on module + title, with fuzzy matching so a reworded title keeps its ID. Retired IDs are never reused. | `state.py` |
| Missing a changed limit | Content hash plus a small tolerance for wording drift, **except** that any changed number always counts as a change. | `state.py` |
| Regenerating everything | Unchanged requirements reuse their cases, verdicts and reviewer edits with **no LLM call**. `--full` forces regeneration. | `pipeline.py` |
| JIRA duplicates | `tc_id → issue key + content hash` is kept in state. A rerun only creates, updates, skips or obsoletes as needed. Regenerated cases inherit the old ID when they match. | `jira_publisher.sync`, `state.reconcile` |
| Stale approvals | Pushing new content voids the approval until the issue is re-approved. If `JIRA_REOPEN_TRANSITION_ID` is set, the issue is moved back to review automatically. | `jira_publisher.pull_reviews` |
| Reviewer edits lost | Edits to Preconditions, Steps or Expected result in JIRA are parsed back into the case and never overwritten. If the requirement later changes, the edit is dropped and the case goes back to review. | `jira_publisher.parse_description` |
| One failure kills the run | Each requirement is retried with backoff and isolated. A failed requirement keeps its previous suite. | `generator._safe` |

## 4. Realistic test data

- **Consolidation:** datasets are joined on shared `*_id`, `*_key` and `*_code` columns (many-to-one). Every account row carries `customers.segment`, `customers.kyc_status` and similar columns, and references can filter on them, e.g. `accounts.account_id[status=ACTIVE,customers.segment=RETAIL]`.
- **Cross-field rules:** constraints such as `amount <= account_id.balance` decide which record is picked, so a limit test is never rejected for insufficient funds by accident. Free fields are adjusted to satisfy the rules. `must_hold=false` builds deliberate violations, e.g. an insufficient-funds case.
- **Setup data:** a daily-limit test gets `_setup.prior_transfer` rows that sum to exactly 9,500 before it sends 500.01.
- **Honest failures:** a rule that can't be satisfied shows up in `_warnings`, and the runner example fails on it.

---

## 5. Project structure

```
genai-testgen-langflow/
├── src/testgen/
│   ├── models.py          # Pydantic schemas = LLM output contracts (cases, constraints, setup, verdicts)
│   ├── prompts.py         # versioned prompts: extract, generate, gap, VERIFY
│   ├── llm.py             # ClaudeLLM (structured outputs); judge model via TESTGEN_JUDGE_MODEL
│   ├── offline.py         # deterministic backend: extraction, limit-aware design, judge
│   ├── ingest.py          # docs loader, data-source join + profiling
│   ├── generator.py       # per-requirement generate → dedupe → verify → gap loop, retries
│   ├── verify.py          # independent judge
│   ├── coverage.py        # verified coverage, gaps, data-aware dedupe
│   ├── state.py           # stable IDs, change detection, test-case ID matching
│   ├── jira_publisher.py  # create/update/obsolete, approvals, reviewer edits, dry-run JIRA
│   ├── testdata.py        # references, joins, cross-field constraints, setup data
│   ├── evaluate.py        # metrics, golden recall, baseline regression gate
│   ├── exporters.py       # outputs + audit log
│   └── pipeline.py        # orchestrator / CLI
├── evals/golden.json      # must-have scenarios for the sample docs
├── langflow_flow/components/test_design/   # 4 Langflow nodes (share state with the CLI)
├── runner_integration/    # pytest consuming generated data
├── samples/docs, samples/data
└── tests/                 # 22 offline tests: data, judge, eval, release workflow
```

---

## 6. Run it

```bash
pip install -e ".[dev]"
```

Real generation with Claude (`ANTHROPIC_API_KEY` must be set):

```bash
python -m testgen.pipeline --docs samples/docs --data samples/data
```

Score the run against the golden set and save it as the baseline (do this once you're happy with a real run):

```bash
python -m testgen.evaluate --update-baseline
```

Later runs, in CI, fail on regression:

```bash
python -m testgen.evaluate
```

The default JIRA is a local dry-run copy in `state/fake_jira.json`. To simulate a reviewer:

```bash
python -m testgen.jira_publisher list
```

```bash
python -m testgen.jira_publisher approve-all
```

Build runner data only for approved, unchanged cases:

```bash
python -m testgen.pipeline --docs samples/docs --data samples/data --approved-only
```

Run the generated data through the runner:

```bash
python -m pytest runner_integration
```

Other flags:
- `--offline` uses the heuristic backend (no API key).
- `--jira` uses real JIRA; set the `JIRA_*` variables from `.env.example`.
- `--full` regenerates every requirement.
- `--no-verify` skips the judge, so coverage becomes self-reported.
- `--state DIR` puts the state somewhere shared.

**State:** `state/state.json` is the memory between releases (IDs, suite, JIRA mapping, reviewer edits). Commit it or keep it on a shared volume. Never mix offline and real runs in the same state folder. Offline cases would be reused as "unchanged", so use `--state` or `--full`.

### Reading a JIRA epic and its user stories

Instead of release-note documents, the pipeline can read an **epic and all its stories**. Each story becomes one requirement whose ID is the **story key**, so test cases are traceable as `PAY-101 → TC-PAY-101-01`.

**Where acceptance criteria are found:**
- a dedicated field (auto-detected by a name containing "Acceptance Criteria", or set `JIRA_AC_FIELD`);
- otherwise an "Acceptance Criteria" section in the description: bullets, numbered items or Given/When/Then;
- stories with no criteria at all are handled by the LLM.

**Change detection works per story:** edit one story and only its test cases are regenerated; JIRA issues are updated, and test cases of removed stories become obsolete. Keep **one state folder per epic**, e.g. `state/PAY-100`.

Demo with the local export (`samples/jira/PAY-100.json`, no JIRA access needed):

```powershell
python -m testgen.pipeline --epic PAY-100 --data samples/data
```

Live JIRA (Cloud or Server/DC; credentials in `.env`):

```powershell
python -m testgen.pipeline --epic PAY-100 --epic-source jira --data samples/data
```

In Langflow, use the **JIRA Epic Reader** node in place of Requirement Extractor and connect its **Requirements** output to Test Case Generator.

### Using Gemini instead of Claude

The provider follows the model name: `gemini-*` models use Gemini, `claude-*` models use Claude. Nothing else changes, including the judge, coverage, test data and JIRA steps.

1. Get a key at Google AI Studio (aistudio.google.com/apikey) and paste it into `.env` as `GEMINI_API_KEY=...`.
2. Check the key works and see which model IDs it can use:

```powershell
.venv\Scripts\python.exe scripts\check_gemini.py
```

3. Run with Gemini from the command line:

```powershell
python -m testgen.pipeline --docs samples/docs --data samples/data --model gemini-3.8-flash
```

In Langflow, pick a `gemini-*` model in the **Model** dropdown on Requirement Extractor and Test Case Generator. You can also type a model ID there.

**Free-tier note:** Gemini's free tier has low per-minute limits. The backend automatically retries rate-limit (429) errors with growing waits, so a full run can take a few minutes. Lower `--workers` if you still hit limits.

## 7. Langflow

Easiest way to start: run `powershell -ExecutionPolicy Bypass -File .\start-langflow.ps1`. It loads your keys from `.env`, registers the custom nodes and starts Langflow. The manual steps are below.

1. Install the package into Langflow's environment:

   ```bash
   pip install langflow
   ```

   ```bash
   pip install -e .
   ```

2. Point Langflow at the components and start it:

   ```bash
   export LANGFLOW_COMPONENTS_PATH="$PWD/langflow_flow/components"
   ```

   ```bash
   langflow run
   ```

   On Windows PowerShell, set the path with `$env:LANGFLOW_COMPONENTS_PATH="$PWD\langflow_flow\components"` instead.
3. A **Test Design** category appears in the sidebar. Wire the nodes like this:

```
Requirement Extractor ──► Test Case Generator ──Test Cases──┬─► JIRA Review Publisher
                                   │                        └─► Synthetic Test Data
                                   └─Coverage Report (verified vs claimed)──► Chat Output
```

The nodes read the same `state/` folder, so IDs match the CLI. The CLI pipeline is what commits state and runs the full release workflow.

**Prototype-only variant (built-in nodes):**
`File` → `Prompt` (`EXTRACT_REQUIREMENTS`) → `Anthropic` → `Structured Output` → `Loop` → `Prompt` (`GENERATE_TEST_CASES`) → `Anthropic` → `Structured Output` → `Prompt` (`VERIFY_TEST_CASES`) → `Anthropic`.

## 8. Running in production

- **Platform swap:** the `LLM` protocol has one method, so an internal GenAI gateway is one small class. The judge model is set separately with `TESTGEN_JUDGE_MODEL`.
- **Data privacy:** only column profiles reach the LLM. Rows come from *test-environment* extracts or are generated.
- **Cost:** system prompts are cached, and unchanged requirements cost nothing on a rerun.
