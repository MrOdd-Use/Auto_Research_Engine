import json
import logging
import os
import time
import warnings
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List

from fastapi import (
    BackgroundTasks,
    FastAPI,
    File,
    HTTPException,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict

from backend.chat.chat import ChatAgentWithMemory
from backend.server.report_store import ReportStore
from backend.server.workflow_store import WorkflowStore
from backend.server.server_utils import (
    execute_multi_agents,
    handle_file_deletion,
    handle_file_upload,
    handle_websocket_communication,
    sanitize_filename,
)
from backend.server.websocket_manager import WebSocketManager, run_agent
from backend.utils import write_md_to_pdf, write_md_to_word
from gpt_researcher.utils.enum import Tone

logger = logging.getLogger(__name__)
logger.propagate = True
logging.getLogger("uvicorn.supervisors.ChangeReload").setLevel(logging.WARNING)

# Models


class ResearchRequest(BaseModel):
    task: str
    report_type: str
    report_source: str
    tone: str
    headers: dict | None = None
    repo_name: str
    branch_name: str
    generate_in_background: bool = True


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="allow")  # Allow extra fields in the request
    
    report: str
    messages: List[Dict[str, Any]]


@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs("outputs", exist_ok=True)
    logger.info("Auto_Research_Engine API ready - local mode (no database persistence)")
    yield
    logger.info("Research API shutting down")

app = FastAPI(lifespan=lifespan)

# Configure allowed origins for CORS
allowed_origins_env = os.getenv("CORS_ALLOW_ORIGINS")
ALLOWED_ORIGINS = (
    [o.strip() for o in allowed_origins_env.split(",") if o.strip()]
    if allowed_origins_env
    else [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://app.gptr.dev",
    ]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("outputs", exist_ok=True)
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

frontend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "frontend"))
static_dir = os.path.join(frontend_dir, "static")

if os.path.isdir(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
else:
    logger.warning("Static assets directory not found: %s", static_dir)

if os.path.isdir(frontend_dir):
    app.mount("/site", StaticFiles(directory=frontend_dir), name="site")
else:
    logger.warning("Frontend directory not found: %s", frontend_dir)

report_store = ReportStore(Path(os.getenv('REPORT_STORE_PATH', os.path.join('data', 'reports.json'))))
workflow_store = WorkflowStore(Path(os.getenv('WORKFLOW_STORE_PATH', os.path.join('data', 'workflows'))))
manager = WebSocketManager(report_store=report_store, workflow_store=workflow_store)
DOC_PATH = os.getenv("DOC_PATH", "./my-docs")

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    index_path = os.path.join(frontend_dir, "index.html")

    if not os.path.exists(index_path):
        raise HTTPException(status_code=404, detail="Frontend index.html not found")

    with open(index_path, "r", encoding="utf-8") as f:
        content = f.read()

    return HTMLResponse(content=content)


@app.get("/report/{research_id}")
async def read_report(research_id: str):
    docx_path = os.path.join('outputs', f"{research_id}.docx")
    if not os.path.exists(docx_path):
        return {"message": "Report not found."}
    return FileResponse(docx_path)


@app.get("/api/reports")
async def get_all_reports(report_ids: str = None):
    report_ids_list = report_ids.split(",") if report_ids else None
    reports = await report_store.list_reports(report_ids_list)
    return {"reports": reports}


@app.get("/api/reports/{research_id}")
async def get_report_by_id(research_id: str):
    report = await report_store.get_report(research_id)
    if report is None:
        raise HTTPException(status_code=404, detail="Report not found")
    return {"report": report}


@app.get("/api/reports/{research_id}/workflow")
async def get_report_workflow(research_id: str, session_id: str | None = None):
    workflow = await workflow_store.build_workflow_response(research_id, session_id=session_id)
    return workflow


@app.post("/api/reports")
async def create_or_update_report(request: Request):
    try:
        data = await request.json()
        research_id = data.get("id", "temp_id")

        now_ms = int(time.time() * 1000)
        existing = await report_store.get_report(research_id)
        incoming_timestamp = data.get("timestamp")
        timestamp = incoming_timestamp if isinstance(incoming_timestamp, int) else now_ms
        if existing and isinstance(existing.get("timestamp"), int):
            timestamp = max(timestamp, existing["timestamp"])

        report = {
            **(existing or {}),
            **{k: v for k, v in data.items() if k != "id" and v is not None},
            "id": research_id,
            "question": data.get("question", (existing or {}).get("question")),
            "answer": data.get("answer", (existing or {}).get("answer", "")),
            "orderedData": data.get("orderedData", (existing or {}).get("orderedData") or []),
            "chatMessages": data.get("chatMessages", (existing or {}).get("chatMessages") or []),
            "timestamp": timestamp,
        }

        await report_store.upsert_report(research_id, report)
        return {"success": True, "id": research_id}
    except Exception as e:
        logger.error(f"Error processing report creation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/api/reports/{research_id}")
async def update_report(research_id: str, request: Request):
    existing = await report_store.get_report(research_id)
    if existing is None:
        raise HTTPException(status_code=404, detail="Report not found")

    data = await request.json()
    now_ms = int(time.time() * 1000)

    updated = {
        **existing,
        **{k: v for k, v in data.items() if v is not None},
        "id": research_id,
        "timestamp": now_ms,
    }

    await report_store.upsert_report(research_id, updated)
    return {"success": True, "id": research_id}


@app.delete("/api/reports/{research_id}")
async def delete_report(research_id: str):
    existed = await report_store.delete_report(research_id)
    if not existed:
        raise HTTPException(status_code=404, detail="Report not found")
    return {"success": True}


@app.get("/api/reports/{research_id}/chat")
async def get_report_chat(research_id: str):
    report = await report_store.get_report(research_id)
    if report is None:
        raise HTTPException(status_code=404, detail="Report not found")
    return {"chatMessages": report.get("chatMessages") or []}


@app.post("/api/reports/{research_id}/chat")
async def add_report_chat_message(research_id: str, request: Request):
    report = await report_store.get_report(research_id)
    if report is None:
        raise HTTPException(status_code=404, detail="Report not found")

    message = await request.json()
    chat_messages = report.get("chatMessages") or []
    if isinstance(chat_messages, list):
        chat_messages = [*chat_messages, message]
    else:
        chat_messages = [message]

    now_ms = int(time.time() * 1000)
    updated = {
        **report,
        "chatMessages": chat_messages,
        "timestamp": now_ms,
    }

    await report_store.upsert_report(research_id, updated)
    return {"success": True, "id": research_id}


async def write_report(research_request: ResearchRequest, research_id: str = None):
    report_result = await run_agent(
        task=research_request.task,
        report_type=research_request.report_type,
        report_source=research_request.report_source,
        source_urls=[],
        document_urls=[],
        tone=Tone[research_request.tone],
        websocket=None,
        stream_output=None,
        headers=research_request.headers,
        query_domains=[],
        config_path="",
    )
    report = (
        str(report_result.get("report", ""))
        if isinstance(report_result, dict)
        else str(report_result)
    )

    docx_path = await write_md_to_word(report, research_id)
    pdf_path = await write_md_to_pdf(report, research_id)
    response = {
        "research_id": research_id,
        "report": report,
        "docx_path": docx_path,
        "pdf_path": pdf_path,
    }

    return response


@app.post("/report/")
async def generate_report(research_request: ResearchRequest, background_tasks: BackgroundTasks):
    if research_request.report_type != "multi_agents":
        raise HTTPException(status_code=400, detail="Only 'multi_agents' report_type is supported.")

    research_id = sanitize_filename(f"task_{int(time.time())}_{research_request.task}")

    if research_request.generate_in_background:
        background_tasks.add_task(write_report, research_request=research_request, research_id=research_id)
        return {"message": "Your report is being generated in the background. Please check back later.",
                "research_id": research_id}
    else:
        response = await write_report(research_request, research_id)
        return response


@app.get("/files/")
async def list_files():
    if not os.path.exists(DOC_PATH):
        os.makedirs(DOC_PATH, exist_ok=True)
    files = os.listdir(DOC_PATH)
    print(f"Files in {DOC_PATH}: {files}")
    return {"files": files}


@app.post("/api/multi_agents")
async def run_multi_agents():
    return await execute_multi_agents(manager)


@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    return await handle_file_upload(file, DOC_PATH)


@app.delete("/files/{filename}")
async def delete_file(filename: str):
    return await handle_file_deletion(filename, DOC_PATH)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        await handle_websocket_communication(websocket, manager)
    except WebSocketDisconnect as e:
        # Disconnect with more detailed logging about the WebSocket disconnect reason
        logger.info(f"WebSocket disconnected with code {e.code} and reason: '{e.reason}'")
        await manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"Unexpected WebSocket error: {str(e)}")
        await manager.disconnect(websocket)


async def _generate_chat_response(report: str, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    chat_agent = ChatAgentWithMemory(
        report=report,
        config_path="default",
        headers=None,
    )
    response_content, tool_calls_metadata = await chat_agent.chat(messages, None)

    if tool_calls_metadata:
        logger.info("Tool calls used: %s", json.dumps(tool_calls_metadata))

    return {
        "role": "assistant",
        "content": response_content,
        "timestamp": int(time.time() * 1000),
        "metadata": {"tool_calls": tool_calls_metadata} if tool_calls_metadata else None,
    }


@app.post("/api/chat")
async def chat(chat_request: ChatRequest):
    try:
        logger.info(f"Received chat request with {len(chat_request.messages)} messages")
        response_message = await _generate_chat_response(
            report=chat_request.report,
            messages=chat_request.messages,
        )
        logger.info("Returning chat response")
        return {"response": response_message}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing chat request: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/api/reports/{research_id}/chat/respond")
async def research_report_chat_response(research_id: str, request: Request):
    """Generate a chat response for a specific report context."""
    try:
        data = await request.json()
        report = data.get("report", "")
        messages = data.get("messages", [])
        if not isinstance(messages, list):
            raise HTTPException(status_code=400, detail="'messages' must be a list")

        response_message = await _generate_chat_response(
            report=report,
            messages=messages,
        )
        return {"response": response_message}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in research report chat: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e
