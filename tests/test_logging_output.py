import pytest
from pathlib import Path
import json
import logging
from fastapi import WebSocket
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MockWebSocket(WebSocket):
    def __init__(self):
        self.events = []
        self.scope = {}

    def __bool__(self):
        return True

    async def accept(self):
        self.scope["type"] = "websocket"
        pass
        
    async def send_json(self, event):
        logger.info(f"WebSocket received event: {event}")
        self.events.append(event)

@pytest.mark.asyncio
async def test_log_output_file():
    """Test to verify logs are properly written to output file"""
    from backend.server.server_utils import CustomLogsHandler
    
    websocket = MockWebSocket()
    await websocket.accept()
    
    query = "What is the capital of France?"
    research_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(query)}"
    logs_handler = CustomLogsHandler(websocket=websocket, task=research_id)

    await logs_handler.send_json({"type": "logs", "content": "info", "output": "Started"})
    await logs_handler.send_json({"query": query, "sources": [], "context": [], "report": "ok"})
    
    logger.info(f"Events captured: {len(websocket.events)}")
    assert len(websocket.events) > 0, "No events were captured"
    
    output_file = Path(logs_handler.log_file)
    output_files = [output_file] if output_file.exists() else []
    assert len(output_files) > 0, "No output file was created"
    
    with open(output_files[-1]) as f:
        data = json.load(f)
        assert len(data.get('events', [])) > 0, "No events in output file" 

    for output_file in output_files:
        output_file.unlink()
        logger.info(f"Deleted output file: {output_file}")
