# Auto_Research_Engine

Auto_Research_Engine is an open-source multi-agent research system for web and local-document research.
It helps you plan tasks, gather sources, synthesize findings, and export structured reports.

## What It Does

- Multi-agent research workflow (planning, retrieval, synthesis, publishing)
- Web + local document research support
- Source-aware report generation (Markdown, optional PDF/DOCX)
- API server for UI and integrations
- CLI for quick report generation
- Optional MCP integration for external tools/data sources

## Tech Stack

- Backend: Python, FastAPI, LangChain/LangGraph
- Frontend: HTML demo + Next.js app (`frontend/nextjs`)
- Retrieval: Tavily and pluggable retrievers
- Output: Markdown, PDF, DOCX

## Quick Start (Local)

### 1) Requirements

- Python 3.11+
- API keys for at least:
  - `OPENAI_API_KEY`
  - `TAVILY_API_KEY`

### 2) Install

```bash
git clone https://github.com/MrOdd-Use/Auto_Research_Engine.git
cd Auto_Research_Engine

# Optional but recommended
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

### 3) Configure Environment

```bash
# copy and edit
cp .env.example .env
```

Set values in `.env`:

```env
OPENAI_API_KEY=your_key
TAVILY_API_KEY=your_key
DOC_PATH=./my-docs
```

### 4) Run Server

```bash
python main.py
```

Server runs at `http://localhost:8000`.

## Run with Docker

```bash
docker compose up --build
```

Default services:

- Backend API: `http://localhost:8000`
- Next.js UI: `http://localhost:3000`

## CLI Usage

Generate a report directly from terminal:

```bash
python cli.py "What are the latest breakthroughs in battery technology?" --tone objective
```

Options:

- `--tone`: `objective|formal|analytical|persuasive|informative|explanatory|descriptive|critical|comparative|speculative|reflective|narrative|humorous|optimistic|pessimistic`
- `--no-pdf`: skip PDF output
- `--no-docx`: skip DOCX output

Reports are saved under `outputs/`.

## Python Usage

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

Note: class name is `GPTResearcher` for code compatibility.

## Research on Local Documents

1. Put files into your docs folder (for example `./my-docs`).
2. Set `DOC_PATH` in `.env`.
3. Use local report source in your flow/config.

Supported formats include PDF, TXT, CSV, Excel, Markdown, PowerPoint, and Word documents.

## Project Structure

```text
backend/            FastAPI server and websocket handling
frontend/           Static and Next.js frontends
gpt_researcher/     Core research logic, retrievers, scrapers
multi_agents/       Multi-agent orchestration workflow
docs/               Documentation site content
tests/              Test suite
```

## Common Commands

```bash
# run API
python main.py

# run tests
python -m pytest

# run docker stack
docker compose up --build
```

## Documentation

- Main docs: https://docs.gptr.dev
- Local docs source: `docs/`

## Contributing

Contributions are welcome. Please read `CONTRIBUTING.md` and open an issue or pull request.

## License

This project is licensed under the MIT License.
