#!/usr/bin/env python3
"""
Auto_Research_Engine Backend Server Startup Script

Run this to start the research API server.
"""

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "backend.server.app:app",
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level="info"
    )


