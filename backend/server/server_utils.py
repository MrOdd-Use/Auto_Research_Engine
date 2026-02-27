import asyncio
import json
import os
import re
import time
import traceback
import unicodedata
from typing import Awaitable, Dict, List, Any
from fastapi.responses import JSONResponse, FileResponse
from gpt_researcher.document.document import DocumentLoader
from gpt_researcher.actions import stream_output
from multi_agents.main import run_research_task
from backend.utils import write_md_to_pdf, write_md_to_word, write_text_to_md
from pathlib import Path
from datetime import datetime
from fastapi import HTTPException
import logging
import hashlib

# Import chat agent
try:
    from backend.chat.chat import ChatAgentWithMemory
except ImportError:
    ChatAgentWithMemory = None

logger = logging.getLogger(__name__)

WINDOWS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    "COM1",
    "COM2",
    "COM3",
    "COM4",
    "COM5",
    "COM6",
    "COM7",
    "COM8",
    "COM9",
    "LPT1",
    "LPT2",
    "LPT3",
    "LPT4",
    "LPT5",
    "LPT6",
    "LPT7",
    "LPT8",
    "LPT9",
}

class CustomLogsHandler:
    """Custom handler to capture streaming logs from the research process"""
    def __init__(self, websocket, task: str):
        self.logs = []
        self.websocket = websocket
        sanitized_filename = sanitize_filename(f"task_{int(time.time())}_{task}")
        self.log_file = os.path.join("outputs", f"{sanitized_filename}.json")
        self.timestamp = datetime.now().isoformat()
        # Initialize log file with metadata
        os.makedirs("outputs", exist_ok=True)
        with open(self.log_file, 'w') as f:
            json.dump({
                "timestamp": self.timestamp,
                "events": [],
                "content": {
                    "query": "",
                    "sources": [],
                    "context": [],
                    "report": "",
                    "costs": 0.0
                }
            }, f, indent=2)

    async def send_json(self, data: Dict[str, Any]) -> None:
        """Store log data and send to websocket"""
        # Send to websocket for real-time display
        if self.websocket:
            await self.websocket.send_json(data)
            
        # Read current log file
        with open(self.log_file, 'r') as f:
            log_data = json.load(f)
            
        # Update appropriate section based on data type
        if data.get('type') == 'logs':
            log_data['events'].append({
                "timestamp": datetime.now().isoformat(),
                "type": "event",
                "data": data
            })
        else:
            # Update content section for other types of data
            log_data['content'].update(data)
            
        # Save updated log file
        with open(self.log_file, 'w') as f:
            json.dump(log_data, f, indent=2)


class Researcher:
    def __init__(self, query: str, report_type: str = "multi_agents"):
        if report_type != "multi_agents":
            raise ValueError("Only 'multi_agents' report_type is supported.")
        self.query = query
        self.report_type = report_type
        # Generate unique ID for this research task
        self.research_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(query)}"
        # Initialize logs handler with research ID
        self.logs_handler = CustomLogsHandler(None, self.research_id)

    async def research(self) -> dict:
        """Conduct research and return paths to generated files"""
        result = await run_research_task(
            query=self.query,
            websocket=self.logs_handler,
            stream_output=stream_output,
        )
        if isinstance(result, dict):
            report = str(result.get("report", ""))
        else:
            report = str(result)
        
        # Generate the files
        sanitized_filename = sanitize_filename(f"task_{int(time.time())}_{self.query}")
        file_paths = await generate_report_files(report, sanitized_filename)
        
        # Get the JSON log path that was created by CustomLogsHandler
        json_relative_path = os.path.relpath(self.logs_handler.log_file)
        
        return {
            "output": {
                **file_paths,  # Include PDF, DOCX, and MD paths
                "json": json_relative_path
            }
        }

def sanitize_filename(filename: str) -> str:
    # Split into components
    prefix, timestamp, *task_parts = filename.split('_')
    task = '_'.join(task_parts)
    task_hash = hashlib.md5(task.encode('utf-8', errors='ignore')).hexdigest()[:10]
            
    # Reassemble and clean the filename
    sanitized = f"{prefix}_{timestamp}_{task_hash}"
    return re.sub(r"[^\w\s-]", "", sanitized).strip()


def secure_filename(filename: str) -> str:
    """
    Sanitize uploaded filenames and reject dangerous inputs.
    """
    if filename is None:
        raise ValueError("Filename is empty")

    normalized = unicodedata.normalize("NFKC", str(filename))
    normalized = normalized.replace("\x00", "")
    normalized = "".join(ch for ch in normalized if ch.isprintable())
    normalized = normalized.strip()

    if not normalized:
        raise ValueError("Filename is empty")

    normalized = re.sub(r"^[A-Za-z]:", "", normalized)

    if re.search(r"(^|[\\/])\.\.([\\/]|$)", normalized):
        raise ValueError("Path traversal detected")

    normalized = normalized.replace("/", "").replace("\\", "")
    normalized = normalized.replace("..", "")
    normalized = re.sub(r"[^\w.\- ]", "", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    normalized = normalized.lstrip(". ").rstrip(". ")

    if not normalized:
        raise ValueError("Filename is empty")

    stem = normalized.split(".", 1)[0].upper()
    if stem in WINDOWS_RESERVED_NAMES:
        raise ValueError("Filename uses reserved name")

    if len(normalized.encode("utf-8")) > 255:
        raise ValueError("Filename is too long")

    return normalized


def validate_file_path(file_path: str, base_dir: str) -> str:
    """
    Ensure file_path resolves inside base_dir after path normalization.
    """
    base_real_path = os.path.realpath(os.path.abspath(base_dir))
    target_real_path = os.path.realpath(os.path.abspath(file_path))

    try:
        common_path = os.path.commonpath([base_real_path, target_real_path])
    except ValueError as exc:
        raise ValueError("Path is outside allowed directory") from exc

    if common_path != base_real_path:
        raise ValueError("Path is outside allowed directory")

    return target_real_path


async def handle_start_command(websocket, data: str, manager):
    json_data = json.loads(data[6:])
    (
        task,
        report_type,
        source_urls,
        document_urls,
        tone,
        headers,
        report_source,
        query_domains,
        mcp_enabled,
        mcp_strategy,
        mcp_configs,
    ) = extract_command_data(json_data)

    if not task or not report_type:
        print("Error: Missing task or report_type")
        await websocket.send_json({
            "type": "logs",
            "content": "error",
            "output": "Missing task or report_type",
        })
        return

    if report_type != "multi_agents":
        await websocket.send_json({
            "type": "logs",
            "content": "error",
            "output": "Only 'multi_agents' report_type is supported.",
        })
        return

    # Create logs handler with websocket and task
    logs_handler = CustomLogsHandler(websocket, task)
    # Initialize log content with query
    await logs_handler.send_json({
        "query": task,
        "sources": [],
        "context": [],
        "report": ""
    })

    sanitized_filename = sanitize_filename(f"task_{int(time.time())}_{task}")

    report = await manager.start_streaming(
        task,
        report_type,
        report_source,
        source_urls,
        document_urls,
        tone,
        websocket,
        headers,
        query_domains,
        mcp_enabled,
        mcp_strategy,
        mcp_configs,
    )
    report = str(report)
    file_paths = await generate_report_files(report, sanitized_filename)
    # Add JSON log path to file_paths
    file_paths["json"] = os.path.relpath(logs_handler.log_file)
    await send_file_paths(websocket, file_paths)


async def handle_human_feedback(data: str):
    feedback_data = json.loads(data[14:])  # Remove "human_feedback" prefix
    print(f"Received human feedback: {feedback_data}")
    # TODO: Add logic to forward the feedback to the appropriate agent or update the research state


def _try_parse_json(payload: str) -> Any:
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return None


def _parse_human_feedback_message(
    raw_data: str, parsed_payload: Any = None
) -> tuple[bool, str | None]:
    """
    Parse supported websocket message formats for human feedback.

    Returns:
        (is_feedback_message, feedback_content)
    """
    if isinstance(parsed_payload, dict) and parsed_payload.get("type") == "human_feedback":
        content = parsed_payload.get("content")
        return True, str(content).strip() if content is not None else None

    stripped = raw_data.strip()
    if not stripped.startswith("human_feedback"):
        return False, None

    trailing = stripped[len("human_feedback"):].strip()
    if not trailing:
        return True, None

    parsed_trailing = _try_parse_json(trailing)
    if isinstance(parsed_trailing, dict):
        content = parsed_trailing.get("content")
        return True, str(content).strip() if content is not None else None
    if parsed_trailing is not None:
        return True, str(parsed_trailing).strip()

    return True, trailing


def _get_or_create_feedback_queue(websocket) -> asyncio.Queue:
    queue = getattr(websocket.state, "human_feedback_queue", None)
    if isinstance(queue, asyncio.Queue):
        return queue

    queue = asyncio.Queue()
    websocket.state.human_feedback_queue = queue
    return queue


def _clear_feedback_queue(websocket) -> None:
    queue = _get_or_create_feedback_queue(websocket)
    while not queue.empty():
        try:
            queue.get_nowait()
        except asyncio.QueueEmpty:
            break


async def _enqueue_human_feedback(websocket, feedback: str | None) -> None:
    queue = _get_or_create_feedback_queue(websocket)
    await queue.put(feedback)


async def handle_chat_command(websocket, data: str):
    """Handle chat command from WebSocket."""
    try:
        # Parse chat data - format is "chat {json_data}"
        json_str = data[5:].strip()  # Remove "chat " prefix
        chat_data = json.loads(json_str)
        
        message = chat_data.get("message", "")
        report = chat_data.get("report", "")
        messages = chat_data.get("messages", [])
        
        # If only message is provided, convert to messages format
        if message and not messages:
            messages = [{"role": "user", "content": message}]
        
        if not messages:
            await websocket.send_json({
                "type": "chat",
                "content": "No message provided.",
                "role": "assistant"
            })
            return
        
        # Check if ChatAgentWithMemory is available
        if ChatAgentWithMemory is None:
            await websocket.send_json({
                "type": "chat",
                "content": "Chat functionality is not available. Please check the server configuration.",
                "role": "assistant"
            })
            return
        
        # Create chat agent with the report context
        chat_agent = ChatAgentWithMemory(
            report=report,
            config_path="default",
            headers=None
        )
        
        # Process the chat
        response_content, tool_calls_metadata = await chat_agent.chat(messages, websocket)
        
        # Send response back via WebSocket
        await websocket.send_json({
            "type": "chat",
            "content": response_content,
            "role": "assistant",
            "metadata": {
                "tool_calls": tool_calls_metadata
            } if tool_calls_metadata else None
        })
        
        logger.info(f"Chat response sent successfully")
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse chat data: {e}")
        await websocket.send_json({
            "type": "chat",
            "content": f"Error: Invalid message format - {str(e)}",
            "role": "assistant"
        })
    except Exception as e:
        logger.error(f"Error handling chat command: {e}\n{traceback.format_exc()}")
        await websocket.send_json({
            "type": "chat",
            "content": f"Error processing your message: {str(e)}",
            "role": "assistant"
        })

async def generate_report_files(report: str, filename: str) -> Dict[str, str]:
    pdf_path = await write_md_to_pdf(report, filename)
    docx_path = await write_md_to_word(report, filename)
    md_path = await write_text_to_md(report, filename)
    return {"pdf": pdf_path, "docx": docx_path, "md": md_path}


async def send_file_paths(websocket, file_paths: Dict[str, str]):
    await websocket.send_json({"type": "path", "output": file_paths})


def get_config_dict(
    langchain_api_key: str, openai_api_key: str, tavily_api_key: str,
    google_api_key: str, google_cx_key: str, bing_api_key: str,
    searchapi_api_key: str, serpapi_api_key: str, serper_api_key: str, searx_url: str
) -> Dict[str, str]:
    return {
        "LANGCHAIN_API_KEY": langchain_api_key or os.getenv("LANGCHAIN_API_KEY", ""),
        "OPENAI_API_KEY": openai_api_key or os.getenv("OPENAI_API_KEY", ""),
        "TAVILY_API_KEY": tavily_api_key or os.getenv("TAVILY_API_KEY", ""),
        "GOOGLE_API_KEY": google_api_key or os.getenv("GOOGLE_API_KEY", ""),
        "GOOGLE_CX_KEY": google_cx_key or os.getenv("GOOGLE_CX_KEY", ""),
        "BING_API_KEY": bing_api_key or os.getenv("BING_API_KEY", ""),
        "SEARCHAPI_API_KEY": searchapi_api_key or os.getenv("SEARCHAPI_API_KEY", ""),
        "SERPAPI_API_KEY": serpapi_api_key or os.getenv("SERPAPI_API_KEY", ""),
        "SERPER_API_KEY": serper_api_key or os.getenv("SERPER_API_KEY", ""),
        "SEARX_URL": searx_url or os.getenv("SEARX_URL", ""),
        "LANGCHAIN_TRACING_V2": os.getenv("LANGCHAIN_TRACING_V2", "true"),
        "DOC_PATH": os.getenv("DOC_PATH", "./my-docs"),
        "RETRIEVER": os.getenv("RETRIEVER", ""),
        "EMBEDDING_MODEL": os.getenv("OPENAI_EMBEDDING_MODEL", "")
    }


def update_environment_variables(config: Dict[str, str]):
    for key, value in config.items():
        os.environ[key] = value


async def handle_file_upload(file, DOC_PATH: str) -> Dict[str, str]:
    os.makedirs(DOC_PATH, exist_ok=True)

    try:
        safe_filename = secure_filename(file.filename)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid file name: {exc}") from exc

    filename_root, filename_ext = os.path.splitext(safe_filename)
    candidate_filename = safe_filename
    candidate_path = validate_file_path(os.path.join(DOC_PATH, candidate_filename), DOC_PATH)

    conflict_index = 1
    while os.path.exists(candidate_path):
        candidate_filename = f"{filename_root}_{conflict_index}{filename_ext}"
        candidate_path = validate_file_path(os.path.join(DOC_PATH, candidate_filename), DOC_PATH)
        conflict_index += 1

    with open(candidate_path, "wb") as buffer:
        file_obj = getattr(file, "file", None)
        if file_obj is None or not hasattr(file_obj, "read"):
            raise HTTPException(status_code=400, detail="Invalid file stream")

        content = file_obj.read()
        if isinstance(content, str):
            content = content.encode("utf-8")
        elif content is None:
            content = b""
        elif not isinstance(content, (bytes, bytearray)):
            content = b""

        buffer.write(bytes(content))
    print(f"File uploaded to {candidate_path}")

    document_loader = DocumentLoader(DOC_PATH)
    await document_loader.load()

    return {"filename": candidate_filename, "path": candidate_path}


async def handle_file_deletion(filename: str, DOC_PATH: str) -> JSONResponse:
    try:
        safe_filename = secure_filename(filename)
        file_path = validate_file_path(os.path.join(DOC_PATH, safe_filename), DOC_PATH)
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"message": f"Invalid file: {exc}"})

    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return JSONResponse(status_code=404, content={"message": "File not found"})

    if not os.path.isfile(file_path):
        return JSONResponse(status_code=400, content={"message": "Path is not a file"})

    os.remove(file_path)
    print(f"File deleted: {file_path}")
    return JSONResponse(content={"message": "File deleted successfully"})


async def execute_multi_agents(manager) -> Any:
    websocket = manager.active_connections[0] if manager.active_connections else None
    if websocket:
        report = await run_research_task("Is AI in a hype cycle?", websocket, stream_output)
        return {"report": report}
    else:
        return JSONResponse(status_code=400, content={"message": "No active WebSocket connection"})


async def handle_websocket_communication(websocket, manager):
    running_task: asyncio.Task | None = None
    _get_or_create_feedback_queue(websocket)

    def run_long_running_task(awaitable: Awaitable) -> asyncio.Task:
        async def safe_run():
            try:
                await awaitable
            except asyncio.CancelledError:
                logger.info("Task cancelled.")
                raise
            except Exception as e:
                logger.error(f"Error running task: {e}\n{traceback.format_exc()}")
                await websocket.send_json(
                    {
                        "type": "logs",
                        "content": "error",
                        "output": f"Error: {e}",
                    }
                )

        return asyncio.create_task(safe_run())

    try:
        while True:
            try:
                data = await websocket.receive_text()
                logger.info(f"Received WebSocket message: {data[:50]}..." if len(data) > 50 else data)

                stripped = data.strip()
                parsed_data = _try_parse_json(stripped) if stripped.startswith("{") else None
                is_feedback_message, feedback_content = _parse_human_feedback_message(
                    stripped, parsed_data
                )

                if data == "ping":
                    await websocket.send_text("pong")
                elif is_feedback_message and running_task and not running_task.done():
                    logger.info("Queued human feedback for active research task")
                    await _enqueue_human_feedback(websocket, feedback_content)
                elif running_task and not running_task.done():
                    # discard any new request if a task is already running
                    logger.warning(
                        f"Received request while task is already running. Request data preview: {data[: min(20, len(data))]}..."
                    )
                    await websocket.send_json(
                        {
                            "type": "logs",
                            "content": "warning",
                            "output": "Task already running. Please wait.",
                        }
                    )
                # Normalize command detection by checking startswith after stripping whitespace
                elif stripped.startswith("start"):
                    logger.info(f"Processing start command")
                    _clear_feedback_queue(websocket)
                    running_task = run_long_running_task(
                        handle_start_command(websocket, data, manager)
                    )
                elif is_feedback_message:
                    logger.info("Queued human feedback without active research task")
                    await _enqueue_human_feedback(websocket, feedback_content)
                elif stripped.startswith("chat"):
                    logger.info(f"Processing chat command")
                    running_task = run_long_running_task(handle_chat_command(websocket, data))
                else:
                    error_msg = f"Error: Unknown command or not enough parameters provided. Received: '{data[:100]}...'" if len(data) > 100 else f"Error: Unknown command or not enough parameters provided. Received: '{data}'"
                    logger.error(error_msg)
                    print(error_msg)
                    await websocket.send_json({
                        "type": "error",
                        "content": "error",
                        "output": "Unknown command received by server"
                    })
            except Exception as e:
                logger.error(f"WebSocket error: {str(e)}\n{traceback.format_exc()}")
                print(f"WebSocket error: {e}")
                break
    finally:
        if running_task and not running_task.done():
            running_task.cancel()

def extract_command_data(json_data: Dict) -> tuple:
    return (
        json_data.get("task"),
        json_data.get("report_type") or "multi_agents",
        json_data.get("source_urls"),
        json_data.get("document_urls"),
        json_data.get("tone"),
        json_data.get("headers", {}),
        json_data.get("report_source"),
        json_data.get("query_domains", []),
        json_data.get("mcp_enabled", False),
        json_data.get("mcp_strategy", "fast"),
        json_data.get("mcp_configs", []),
    )
