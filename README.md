# Auto_Research_Engine

Auto_Research_Engine is an open-source multi-agent deep research system for both web and local-document workflows. It turns task understanding, information retrieval, evidence organization, report writing, and result publishing into an orchestrated, observable, and extensible engineering pipeline. It is well suited for industry analysis, technical research, knowledge synthesis, report generation, and research copilot scenarios.

The core idea is not to let a single model generate a long answer in one shot. Instead, the system breaks research into explicit stages handled by specialized modules and agents: plan first, then retrieve; verify first, then write; draft first, then review; and finally export and interact with the result.

## Core Capabilities

- Supports both web research and local document research
- Multi-agent state-machine workflow built on LangGraph
- Section-level parallel deep research and result aggregation
- Node rerun support (`节点回溯` / `Rerun from Checkpoint`) for the `multi_agents` workflow
- Multi-format export to Markdown, PDF, and DOCX
- FastAPI API, WebSocket streaming, CLI, Python SDK, and frontend interfaces
- Optional MCP integration with pluggable retrievers, scrapers, and model providers
- Built-in evaluations, tests, documentation, and deployment infrastructure

## Tech Stack

- Backend: Python, FastAPI, WebSocket
- Orchestration: LangGraph, LangChain
- Research engine: pluggable retrievers, scrapers, vector retrieval, context compression
- Frontend: static HTML/CSS/JS and Next.js/React
- Engineering: Docusaurus, Pytest, Docker, Terraform

## End-to-End Workflow

1. A user submits a research task from the CLI, HTTP API, WebSocket, or frontend.
2. The multi-agent orchestration layer performs initial research and creates a structured outline.
3. A human-in-the-loop step can inject feedback before deep research begins, reducing direction drift.
4. Each section enters the research pipeline in parallel and calls retrievers, scrapers, and context-management modules to collect evidence.
5. The `check_data` quality gate decides whether the current evidence is sufficient for that section or whether more research is needed.
6. The Writer aggregates section outputs and generates the introduction, table of contents, body, conclusion, and references.
7. Reviewer and Reviser agents refine the final draft until it is publishable or reaches the review cap.
8. The backend exports Markdown, PDF, and DOCX, then exposes the result through report storage, chat endpoints, and frontend interfaces.

## Node Rerun

The `multi_agents` workflow now supports `节点回溯` (`Rerun from Checkpoint`) so users can selectively rerun part of a completed report instead of starting over.

- Each initial run or rerun creates a new workflow session under `data/workflows/{report_id}/`
- Global workflow nodes such as `browser`, `planner`, `researcher`, `writer`, `reviewer`, `reviser`, and `publisher` are recorded as rerunnable checkpoints
- Section-level checkpoints are also recorded inside the researcher stage, including `scrap` and `check_data` when ASA is enabled
- A rerun can restart from a global node or a single section node and will only recompute the required downstream path
- Multiple rounds of changes are preserved as parent/child sessions so a report keeps its full revision history

## Module Breakdown

### 1. Entry and Runtime Layer

| Module | Role | Notes |
| --- | --- | --- |
| `main.py` | Application entry point | Loads environment variables, initializes logging, and starts the FastAPI service. |
| `cli.py` | Command-line interface | Generates research reports from the terminal and fits scripting or automation use cases. |
| `docker-compose.yml` / `Dockerfile*` | Containerized deployment | Provides standardized local and full-stack runtime options. |
| `.env.example` | Environment template | Centralizes LLM, retriever, document path, tracing, and runtime settings. |

### 2. Core Research Engine `gpt_researcher/`

`gpt_researcher/` is the core capability layer of the system. It is responsible for turning a natural-language task into an executable research process.

| Submodule | Role | Key Capabilities |
| --- | --- | --- |
| `agent.py` | Research controller | Defines `GPTResearcher` and coordinates retrieval, scraping, context management, report writing, image planning, and MCP strategy. |
| `actions/` | Research actions | Handles query processing, search orchestration, report generation, markdown processing, and agent creation. |
| `skills/` | High-level research skills | Includes research execution, browser management, context management, source curation, deep research, image generation, and writing. |
| `retrievers/` | Retrieval adapters | Supports Tavily, DuckDuckGo, Google, Bing, Serper, SearchAPI, Searx, Semantic Scholar, Arxiv, PubMed Central, MCP, and more. |
| `scraper/` | Content extraction layer | Provides browser-based scraping, BeautifulSoup, PyMuPDF, Firecrawl, Tavily Extract, Arxiv, and related extractors. |
| `context/` | Context processing | Compresses, retrieves, and assembles context from multiple sources to reduce long-context noise. |
| `memory/` / `vector_store/` | Memory and vector retrieval | Handles embeddings, vector store wrapping, and similarity retrieval for document-heavy research. |
| `document/` | Document ingestion | Unifies online documents, local documents, and Azure-backed document loading flows. |
| `llm_provider/` | Model provider abstraction | Keeps model calling logic provider-agnostic so the system can switch or extend LLM vendors cleanly. |
| `mcp/` | MCP integration | Manages tool selection, streaming, and MCP-aware research flows. |
| `config/` | Configuration management | Centralizes defaults, variable injection, and runtime configuration. |
| `utils/` | Shared infrastructure | Includes logging, rate limiting, cost tracking, validation, tools, and enums. |

The value of this layer is that it is not just a thin wrapper around LLM calls. It abstracts a complete retrieve-scrape-compress-write research loop that can be reused both by single-task workflows and by the multi-agent system.

### 3. Multi-Agent Orchestration Layer `multi_agents/`

`multi_agents/` breaks research into a long-running collaborative workflow across multiple roles. This is one of the main differences between this system and a standard RAG application.

| Module / Role | Role | Notes |
| --- | --- | --- |
| `main.py` | Multi-agent entry point | Loads task settings from `task.json` and launches the research team. |
| `agents/orchestrator.py` | Chief orchestrator | Uses `ChiefEditorAgent` and LangGraph to build the state graph that connects browser, planner, human, researcher, writer, reviewer, reviser, and publisher. |
| `agents/editor.py` | Planning and parallel scheduling | Produces structured outlines in `section_details` and starts section-level parallel research and validation. |
| `agents/researcher.py` / `agents/scrap.py` | Section research execution | Calls the underlying research engine to gather evidence and generate section drafts or research packets. |
| `agents/check_data.py` | Evidence gate | Decides whether section evidence is acceptable, should retry, or should be blocked. |
| `agents/writer.py` | Report composition | Builds the final report structure, including introduction, table of contents, conclusion, and unified body content. |
| `agents/reviewer.py` | Final review | Evaluates the final draft from a publishability and quality perspective. |
| `agents/reviser.py` | Final revision | Revises the final draft based on reviewer feedback until accepted or capped. |
| `agents/publisher.py` | Output publishing | Organizes report artifacts and exports them to multiple formats. |
| `agents/human.py` | Human feedback node | Accepts user feedback at key workflow checkpoints. |
| `memory/` | Workflow state definitions | Uses structures like `ResearchState` and `DraftState` to track global and section-level state. |

This layer is not mainly about making the model "smarter". It is about making complex research more stable through explicit state, conditional routing, and controlled feedback loops.

### 4. Service and Interaction Layer

#### `backend/`

`backend/` exposes the research system as a runnable service layer and turns the orchestration flow into a usable API product surface.

| Module | Role | Notes |
| --- | --- | --- |
| `backend/server/app.py` | FastAPI app | Exposes report CRUD, chat, file upload, and WebSocket endpoints. |
| `backend/server/websocket_manager.py` | Streaming session manager | Manages active connections, message queues, and live task output for real-time frontend updates. |
| `backend/server/server_utils.py` | Service helpers | Handles command parsing, uploads, file deletion, log persistence, file export, and human-feedback queuing. |
| `backend/server/report_store.py` | Report persistence | Uses a lightweight JSON-backed store with atomic temp-file replacement for report data. |
| `backend/server/workflow_store.py` | Workflow-session persistence | Stores checkpoint trees, session lineage, selected rerun targets, and last successful workflow state for `Rerun from Checkpoint`. |
| `backend/chat/chat.py` | Report chat agent | Lets users keep asking questions about an existing report and optionally trigger `quick_search` when fresh information is needed. |
| `backend/utils.py` | Export utilities | Converts Markdown output into `.md`, `.pdf`, and `.docx`. |
| `backend/memory/` | Runtime state data | Stores intermediate research and draft-related runtime data for backend flows. |

#### `frontend/`

`frontend/` provides two UI forms: a lightweight demo frontend and a more complete Next.js-based application.

| Module | Role | Notes |
| --- | --- | --- |
| `frontend/index.html` + `scripts.js` + `styles.css` | Static frontend | Can be served directly by the backend and is useful for quick demos and local exploration. |
| `frontend/nextjs/` | Next.js app and UI package | Provides a richer research interface, report views, research history, `节点回溯` (`Rerun from Checkpoint`) controls, mobile layouts, settings panels, and embeddable React components. |
| `frontend/nextjs/hooks/` | Frontend data layer | Contains WebSocket handling, research history, scroll behavior, and analytics helpers. |
| `frontend/nextjs/components/` | Interaction components | Includes research forms, log panels, chat panels, report blocks, image displays, settings components, and the workflow-session panel used for `Rerun from Checkpoint`. |

### 5. Supporting Engineering Modules

| Module | Role | Notes |
| --- | --- | --- |
| `docs/` | Documentation site | Uses Docusaurus for docs, examples, blogs, and reference content. |
| `evals/` | Evaluation framework | Contains factuality and hallucination evaluations for offline quality checks. |
| `tests/` | Automated tests | Covers orchestrator flow, logging, retrieval, MCP, and security-related paths. |
| `terraform/` | Infrastructure templates | Provides cloud and CI-related infrastructure setup templates. |
| `mcp-server/` | MCP server notes | Points to the dedicated MCP repository for external tool and data-source integration. |
| `questions/` | Project explanation materials | Contains architecture and interview-style project explanation notes aligned with the implementation. |

## Technical Highlights

### 1. Complex research is driven by a state machine, not a single giant prompt

The system models research as an explicit workflow using LangGraph instead of relying on one long prompt. That makes the route between planning, human feedback, deep research, review, revision, and publishing much more controllable and maintainable.

### 2. Section-level parallel deep research balances speed and quality

The planner first creates structured `section_details`, then each section is researched and validated independently before aggregation. This is more scalable than a single serial chain and better suited to long-form reports and multi-topic analysis.

### 3. Web and local-document research share one unified pipeline

The system can work with web search, web scraping, and local sources such as PDF, Word, Markdown, and Excel. That makes it useful not only for public-information research but also for hybrid workflows that combine external sources with internal documents.

### 4. Retrievers, scrapers, and model providers are all pluggable

Retrievers, scrapers, and LLM providers are implemented as adapter layers. That means new search sources, new extraction backends, or new model vendors can be added without rewriting the entire research workflow.

### 5. Two-layer quality control: evidence gating plus final review

The system does not go straight from retrieved content to final writing. It adds a section-level `check_data` evidence gate and a final Reviewer/Reviser loop. The first asks whether the evidence is sufficient; the second asks whether the draft is publishable. This is much more structured than basic retrieve-then-generate pipelines.

### 6. Real-time streaming and observability are built into the product surface

The backend pushes logs, stage updates, and output paths to the frontend over WebSocket, while also writing research events into JSON logs. That makes the process visible instead of turning research into a black-box generation step.

### 7. Reports remain interactive after generation

`backend/chat/chat.py` allows users to continue asking questions about a finished report. The system can answer from report context or trigger tool-assisted search when fresh information is required. This makes the report itself a living context object rather than a dead file.

### 8. The codebase shows product-oriented engineering, not just experimentation

The system includes filename sanitization, path validation, report persistence with locking and atomic replacement, and multi-format delivery. Those choices show that the repository is moving toward production-minded engineering rather than staying at the notebook-demo stage.

### 9. Evaluation, testing, docs, and deployment are part of the repository

The repository includes `evals/`, `tests/`, Docker files, Terraform templates, and a documentation site. That gives it a fuller engineering lifecycle than a project that only ships core model logic.

## Quick Start

### 1. Requirements

- Python 3.11+
- At least one working LLM provider and one retrieval provider
- Recommended minimum setup:
  - `OPENAI_API_KEY`
  - `TAVILY_API_KEY`

### 2. Install

```bash
git clone https://github.com/MrOdd-Use/Auto_Research_Engine.git
cd Auto_Research_Engine

python -m venv .venv

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

### 3. Configure Environment

```bash
cp .env.example .env
```

Example:

```env
OPENAI_API_KEY=your_key
TAVILY_API_KEY=your_key
DOC_PATH=./my-docs
```

### 4. Run the Server

```bash
python main.py
```

Default endpoints:

- API: `http://localhost:8000`
- Static frontend: `http://localhost:8000/`

### 5. Run with Docker

```bash
docker compose up --build
```

Default services:

- Backend API: `http://localhost:8000`
- Next.js UI: `http://localhost:3000`

## Usage

### CLI

```bash
python cli.py "What are the latest breakthroughs in battery technology?" --tone objective
```

Common options:

- `--tone`: control output tone
- `--no-pdf`: skip PDF export
- `--no-docx`: skip DOCX export

### Python

```python
import asyncio
from gpt_researcher import GPTResearcher

async def run():
    researcher = GPTResearcher(query="Why is Nvidia stock going up?")
    await researcher.conduct_research()
    report = await researcher.write_report()
    print(report)

asyncio.run(run())
```

### Local Document Research

1. Put your files into a local docs folder such as `./my-docs`
2. Set `DOC_PATH` in `.env`
3. Use the local or hybrid source mode in your workflow/configuration

Supported formats include PDF, TXT, CSV, Excel, Markdown, PowerPoint, and Word documents.

### Workflow APIs

For the Next.js frontend and custom integrations, the workflow layer exposes:

- `GET /api/reports/{id}/workflow` to fetch sessions and the checkpoint tree for a report
- WebSocket `start` with `report_id` to create a stable workflow-linked report
- WebSocket `rerun` with `report_id`, `checkpoint_id`, and optional `note` to trigger `Rerun from Checkpoint`

## Project Structure

```text
backend/            FastAPI service, WebSocket flow, chat APIs, report storage and export
frontend/           Static frontend and Next.js frontend
gpt_researcher/     Core research engine, retrievers, scrapers, context management, model adapters
multi_agents/       LangGraph-based multi-agent orchestration workflow
docs/               Documentation site source
evals/              Factuality and hallucination evaluations
tests/              Automated test suite
terraform/          Infrastructure deployment templates
questions/          Project explanation and architecture notes
```

## Common Commands

```bash
# Start API
python main.py

# Run tests
python -m pytest

# Start full Docker stack
docker compose up --build
```

## Documentation and Extensions

- Official docs: https://docs.gptr.dev
- Frontend notes: `frontend/`
- Multi-agent notes: `multi_agents/README.md`
- Evaluation notes: `evals/README.md`
- MCP notes: `mcp-server/README.md`

## License

This project is licensed under the MIT License.
