# Repository Structure

This document describes the current repository layout and the intended ownership of each major area.

## Core Modules

- `gpt_researcher/`
  - Core research engine and reusable libraries.
  - Includes retrievers, scrapers, context management, config, and model providers.

- `backend/`
  - API and runtime surface for local/server usage.
  - `backend/server/` contains FastAPI routes, websocket orchestration, and report persistence.
  - `backend/utils.py` contains shared export utilities (Markdown, PDF, DOCX).

- `multi_agents/`
  - Multi-agent orchestration flow (editor/researcher/check_data/writer/claim_verifier/reviewer/reviser/publisher).
  - `workflow_session.py` and backend workflow storage persist rerunnable global and section checkpoints.
  - Uses the same backend export styling for PDF generation.

- `frontend/`
  - `frontend/index.html` static frontend served by the backend.
  - `frontend/nextjs/` production-oriented frontend application.

## Route and API Notes

- Report chat history persistence:
  - `POST /api/reports/{research_id}/chat`
- Report-context response generation:
  - `POST /api/reports/{research_id}/chat/respond`
- General chat endpoint:
  - `POST /api/chat`

## Shared Asset Rule

- Canonical PDF stylesheet path:
  - `backend/styles/pdf_styles.css`
- Other components should reference this file instead of carrying duplicate copies.
