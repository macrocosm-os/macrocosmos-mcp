"""
Macrocosmos MCP Server - SSE with Bearer token auth
Based on working Bittensor pattern - just converted tools and auth
"""

from __future__ import annotations

import contextvars
import json
import os
from typing import List, Optional

import macrocosmos as mc
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from mcp.server.fastmcp import FastMCP
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("macrocosmos_mcp_sse")

# ---------------------------------------------------------------------------
# Context propagation
# ---------------------------------------------------------------------------

request_var: contextvars.ContextVar[Request] = contextvars.ContextVar("request")

# ---------------------------------------------------------------------------
# FastMCP + FastAPI setup
# ---------------------------------------------------------------------------

mcp = FastMCP("macrocosmos", request_timeout=300)
app = FastAPI(title="Macrocosmos MCP Server")

bearer_scheme = HTTPBearer(auto_error=False)

def get_user_api_key() -> str:
    """Get current user's API key from request context"""
    try:
        request = request_var.get()
        return getattr(request.state, 'user_api_key', '')
    except:
        return os.getenv("MC_API", "")

async def verify_token(request: Request) -> None:
    """Validate Bearer token (user's Macrocosmos API key)."""
    credentials: HTTPAuthorizationCredentials | None = await bearer_scheme(request)
    token = credentials.credentials if credentials else None

    if not token:
        raise HTTPException(status_code=401, detail="Missing Bearer token")

    # Store user's API key in request state
    request.state.user_api_key = token
    request_var.set(request)

@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    """Block every request that doesn't carry a valid Bearer token."""
    try:
        await verify_token(request)
    except HTTPException as exc:
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
    return await call_next(request)

# Mount MCP after the middleware so SSE handshakes are protected too
app.mount("/", mcp.sse_app())

# ---------------------------------------------------------------------------
# MCP Tools
# ---------------------------------------------------------------------------

@mcp.tool(description="""
Fetch real-time social media data from X (Twitter) and Reddit through the Macrocosmos SN13 network.

IMPORTANT: This tool requires 'source' parameter to be either 'X' or 'REDDIT' (case-sensitive).

Parameters:
- source (str, REQUIRED): Data platform - must be 'X' or 'REDDIT'
- usernames (List[str], optional): Up to 5 Twitter usernames to monitor (X only - NOT available for Reddit). Each username must start with '@' (e.g., ['@elonmusk', '@sundarpichai'])
- keywords (List[str], optional): Up to 5 keywords/hashtags to search. For Reddit, use subreddit names (e.g., ['MachineLearning', 'technology'])
- start_date (str, optional): Start timestamp in ISO format (e.g., '2024-01-01T00:00:00Z'). Defaults to 24h ago if not specified
- end_date (str, optional): End timestamp in ISO format (e.g., '2024-06-03T23:59:59Z'). Defaults to current time if not specified  
- limit (int, optional): Maximum results to return (1-1000). Default: 10

Usage Examples:
1. Monitor Twitter users: query_on_demand_data(source='X', usernames=['@elonmusk', '@sundarpichai'], limit=20)
2. Search Twitter keywords: query_on_demand_data(source='X', keywords=['AI', '#MachineLearning'], limit=50)
3. Monitor Reddit subreddits: query_on_demand_data(source='REDDIT', keywords=['MachineLearning', 'technology'], limit=30)
4. Time-bounded search: query_on_demand_data(source='X', keywords=['Bitcoin'], start_date='2024-06-01T00:00:00Z', end_date='2024-06-03T23:59:59Z')

Returns: Structured data with content, metadata, user info, timestamps, and platform-specific details. Each item includes URI, datetime, source, label, content preview, and additional metadata.
""")
async def query_on_demand_data(
    source: str,
    usernames: Optional[List[str]] = None,
    keywords: Optional[List[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: int = 10
) -> str:
    """Query data on demand from various sources."""

    user_api_key = get_user_api_key()

    if not user_api_key:
        return "Error: No Macrocosmos API key available"

    client = mc.AsyncSn13Client(api_key=user_api_key)

    response = await client.sn13.OnDemandData(
        source=source,
        usernames=usernames if usernames else [],
        keywords=keywords if keywords else [],
        start_date=start_date,
        end_date=end_date,
        limit=limit
    )

    if not response:
        return "Failed to fetch data. Please check your API key and parameters."

    # Convert response to dict if it's a Pydantic model or similar object
    if hasattr(response, 'model_dump'):
        response_dict = response.model_dump()
    elif hasattr(response, 'dict'):
        response_dict = response.dict()
    elif isinstance(response, dict):
        response_dict = response
    else:
        response_dict = {"data": str(response)}

    return json.dumps(response_dict, indent=2, default=str)

# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    import uvicorn

    port = int(os.getenv("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    main()