<div align="center">
  <h1>Auto_Research_Engine</h1>

  <p>
    Open-source deep research for web and local-document workflows.
    Plan, retrieve, validate, write, review, and publish in one system.
  </p>

</div>

Auto_Research_Engine is a multi-agent research system for generating long-form, sourced reports from web data, local documents, or both. Instead of asking one model to produce a large answer in a single pass, it breaks the job into explicit stages: planning, evidence collection, validation, writing, review, revision, and publishing.

It is built for research-heavy workflows such as market analysis, technical investigations, company briefs, internal knowledge synthesis, and report-generation products.

## Why Auto_Research_Engine

- **Multi-agent by default.** The system separates planning, retrieval, evidence gating, writing, review, and publishing into specialized agents.
- **Web and document research.** Use search, scraping, local files, or hybrid workflows in the same pipeline.
- **Section-level parallelism.** Reports are decomposed into sections that can be researched independently and then merged.
- **Source-grounded final drafting.** Writer consumes an append-only `source_index`, emits `claim_annotations`, and gives each factual claim explicit `[S*]` citations.
- **Built-in claim verification.** `ClaimVerifier` grades claims as `HIGH` / `MEDIUM` / `SUSPICIOUS` / `HALLUCINATION` and can trigger targeted section reruns for suspicious evidence conflicts.
- **Node rerun support.** `节点回溯` (`Rerun from Checkpoint`) lets you rerun a single workflow node or section instead of restarting the entire report.
- **Production-facing interfaces.** The repo includes FastAPI, WebSocket streaming, a Next.js frontend, CLI usage, exports, tests, and docs.
- **Pluggable architecture.** Retrievers, scrapers, model providers, vector stores, and MCP integrations can be swapped without redesigning the whole stack.
- **Intent-aware pipeline.** `IntentRecognizer` classifies query intent (`analytical`, `descriptive`, `comparative`, `exploratory`) and normalizes the query before planning, so the outline and research strategy reflect what the user actually wants to understand.
- **Section-level synthesis.** `SectionSynthesizer` converts accepted evidence passages into structured markdown prose with sub-headers and inline `[Sx]` citations, keeping every factual claim grounded before the global writer runs.
- **Persistent opinion tracking.** `OpinionsStore` accumulates reviewer and human opinions across multiple review rounds and tracks resolution status (`pending` → `resolved` / `unresolved` / `partially_resolved`), so no feedback item is silently dropped.
- **Human review breakpoints.** The pipeline pauses at configurable checkpoints — outline review, reviewer feedback, and writer output — for human input before proceeding.
- **Scraper fallback chain.** The scraper tries multiple backends in sequence and falls back gracefully when a source is unreachable or context is exceeded.

## How It Works

1. A user submits a research task through the CLI, API, WebSocket, or frontend.
2. `IntentRecognizer` classifies query intent, normalizes the query, and produces a research goal. This output propagates through the entire pipeline.
3. The planner creates a structured outline from the task and the initial research context.
4. Each section runs through its own research pipeline in parallel.
5. `check_data` decides whether the evidence for a section is sufficient, should retry, or should stop.
6. `SectionSynthesizer` converts accepted evidence passages into structured markdown prose with sub-headers and `[Sx]` citations, building a per-section local source index that is later remapped to the global `source_index`.
7. The writer composes the introduction and conclusion from indexed section evidence and emits `claim_annotations`.
8. `ClaimVerifier` parses citations, checks support across domains, and can trigger targeted `Reflexion` reruns for suspicious sections.
9. `OpinionsStore` accumulates reviewer-agent and human opinions across review rounds, tracking each item's resolution status.
10. Reviewer and reviser agents run a final quality loop with guideline checks plus source-aware hallucination auditing before the publisher exports Markdown, PDF, and DOCX.

## Node Rerun

The `multi_agents` workflow supports `节点回溯` (`Rerun from Checkpoint`).

- Every initial run or rerun creates a new workflow session tied to the same `report_id`
- Global nodes such as `browser`, `planner`, `researcher`, `writer`, `reviewer`, `reviser`, and `publisher` are stored as rerunnable checkpoints
- Section-level checkpoints are also stored, including `scraping` and `check_data` when ASA is enabled
- A rerun only recomputes the selected node and its downstream path
- Multi-round edits are preserved as parent/child sessions, so the latest failed rerun does not overwrite the last successful result

Workflow session data is stored under:

```text
data/workflows/{report_id}/
```

## Interfaces

| Interface | Use case | Entry point |
| --- | --- | --- |
| CLI | Generate a report from the terminal | `python cli.py "<query>" --tone objective` |
| FastAPI | CRUD, chat, uploads, workflow APIs | `backend/server/app.py` |
| WebSocket | Streaming research output and reruns | `/ws` |
| Next.js frontend | Research UI, report history, `Rerun from Checkpoint` | `frontend/nextjs/` |
| Python package | Integrate the research engine in code | `gpt_researcher/` |

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+ if you want the Next.js frontend
- At least one LLM provider and one retrieval provider

Recommended minimum environment:

- `OPENAI_API_KEY`
- `TAVILY_API_KEY`

### 1. Clone and install

```bash
git clone https://github.com/MrOdd-Use/Auto_Research_Engine.git
cd Auto_Research_Engine

python -m venv .venv

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
```

Minimum example:

```env
OPENAI_API_KEY=your_key
TAVILY_API_KEY=your_key
DOC_PATH=./my-docs
```

### 3. Run the backend

```bash
python main.py
```

Default backend URL:

- `http://localhost:8000`

### 4. Run the Next.js frontend

```bash
cd frontend/nextjs
npm install --legacy-peer-deps
npm run dev
```

Default frontend URL:

- `http://localhost:3000`

You can also run the full stack with Docker:

```bash
docker compose up --build
```

## Usage

### CLI

```bash
python cli.py "What are the latest breakthroughs in battery technology?" --tone objective
```

### Workflow API

The workflow layer used by the frontend exposes:

- `GET /api/reports/{id}/workflow` to fetch workflow sessions and checkpoint trees
- WebSocket `start` with `report_id` to create a stable workflow-linked report
- WebSocket `rerun` with `report_id`, `checkpoint_id`, and optional `note` to trigger `Rerun from Checkpoint`

### Local document research

1. Put your files into a local folder such as `./my-docs`
2. Set `DOC_PATH` in `.env`
3. Use the local or hybrid source mode in your workflow

Supported formats include PDF, TXT, CSV, Excel, Markdown, PowerPoint, and Word documents.

## Repository Structure

```text
backend/          FastAPI app, WebSocket handling, report and workflow persistence
docs/             Documentation site
evals/            Evaluation scripts and datasets
frontend/         Static frontend and Next.js frontend
gpt_researcher/   Core research engine
multi_agents/     Multi-agent orchestration, writer/reviewer/reviser flow
tests/            Automated tests
```

## Project Highlights

- `gpt_researcher/` contains the core retrieve-scrape-compress-write engine
- `multi_agents/` adds planning, section-level research, append-only source indexing, claim verification, and final review loops
- `backend/server/workflow_store.py` persists workflow sessions for `Rerun from Checkpoint`
- `frontend/nextjs/` provides report history, session switching, and rerun controls
- `tests/test_workflow_sessions.py` covers multi-round reruns and checkpoint behavior
- `multi_agents/agents/intent_recognizer.py` classifies query intent and normalizes the query before planning
- `multi_agents/agents/section_synthesizer.py` converts evidence passages into structured section bodies with sub-headers and citations
- `multi_agents/memory/opinions.py` accumulates and tracks reviewer and human opinions across review rounds

## Documentation

- Docs: https://docs.gptr.dev
- Frontend guide: `docs/docs/gpt-researcher/frontend/nextjs-frontend.md`
- Multi-agent guide: `multi_agents/README.md`

## Contributing

Contributions are welcome. If you want to contribute:

1. Open an issue for bugs, regressions, or feature proposals
2. Fork the repo and create a focused branch
3. Add tests for behavior changes where possible
4. Open a pull request with a clear summary of the change

## Community

- Discord: https://discord.gg/QgZXvJAccX
- Issues: https://github.com/MrOdd-Use/Auto_Research_Engine/issues

## License

MIT
