"""
Macrocosmos MCP Server - SSE with Bearer token auth
Based on working Bittensor pattern - just converted tools and auth
"""

from __future__ import annotations

import asyncio
import contextvars
import os
from typing import Any, Dict, List, Optional

import httpx
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
# Configuration
# ---------------------------------------------------------------------------

APEX_BASE_URL = "https://constellation.api.cloud.macrocosmos.ai"

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
# Helper Classes
# ---------------------------------------------------------------------------

class ApexClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = APEX_BASE_URL
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    async def chat_completion(
            self,
            messages: List[Dict[str, str]],
            temperature: float = 0.7,
            top_p: float = 0.95,
            max_new_tokens: int = 256,
            do_sample: bool = True,
            model: str = "Default"
    ) -> Dict[str, Any]:
        payload = {
            "messages": messages,
            "sampling_parameters": {
                "temperature": temperature,
                "top_p": top_p,
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample
            }
        }

        if model != "Default":
            payload["model"] = model

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{self.base_url}/apex.v1.ApexService/ChatCompletion",
                headers=self.headers,
                json=payload
            )
            response.raise_for_status()
            return response.json()

    async def web_search(
            self,
            search_query: str,
            n_miners: int = 3,
            max_results_per_miner: int = 2,
            max_response_time: int = 30
    ) -> Dict[str, Any]:
        payload = {
            "search_query": search_query,
            "n_miners": n_miners,
            "max_results_per_miner": max_results_per_miner,
            "max_response_time": max_response_time
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{self.base_url}/apex.v1.ApexService/WebRetrieval",
                headers=self.headers,
                json=payload
            )
            response.raise_for_status()
            return response.json()

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

    status = response.get("status")

    if status == "error":
        error_msg = response.get("meta", {}).get("error", "Unknown error")
        return f"Error: {error_msg}"

    data = response.get("data", [])
    meta = response.get("meta", {})

    if not data:
        return "No data found for the specified criteria."

    formatted_data = []
    for item in data:
        formatted_item = f"""
            URI: {item.get('uri', 'N/A')}
            Date: {item.get('datetime', 'N/A')}
            Source: {item.get('source', 'N/A')}
            Label: {item.get('label', 'N/A')}
            Content: {item.get('content', 'No content')[:200]}...
            MetaData: {item.get('tweet', 'N/A')}
            UserInfo: {item.get('user', 'N/A')}
            Media: {item.get('media', 'No media')}
        """
        formatted_data.append(formatted_item)

    meta_info = f"""
    Meta Information:
        - Miners queried: {meta.get('miners_queried', 'N/A')}
        - Data source: {meta.get('source', 'N/A')}
        - Items returned: {meta.get('items_returned', len(data))}
    """

    return meta_info + "\n\n" + "\n---\n".join(formatted_data)


@mcp.tool()
async def apex_chat(
        prompt: str,
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_new_tokens: int = 256,
        do_sample: bool = True,
        model: str = "Default"
) -> str:
    """Send a chat completion request to Apex (SN1) decentralized LLMs."""

    user_api_key = get_user_api_key()

    if not user_api_key:
        return "Error: No Macrocosmos API key available"

    try:
        client = ApexClient(user_api_key)
        messages = [{"role": "user", "content": prompt}]

        result = await client.chat_completion(
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            model=model
        )

        if "choices" in result and len(result["choices"]) > 0:
            response_content = result["choices"][0]["message"]["content"]
            model_used = result.get("model", "Unknown")
            return f"**Model:** {model_used}\n\n**Response:**\n{response_content}"
        else:
            return f"Error: Unexpected response format: {result}"

    except Exception as e:
        return f"Error: {str(e)}"


@mcp.tool()
async def apex_web_search(
        search_query: str,
        n_miners: int = 3,
        max_results_per_miner: int = 2,
        max_response_time: int = 30
) -> str:
    """Perform web search using Apex's decentralized web retrieval."""

    user_api_key = get_user_api_key()

    if not user_api_key:
        return "Error: No Macrocosmos API key available"

    try:
        client = ApexClient(user_api_key)

        result = await client.web_search(
            search_query=search_query,
            n_miners=n_miners,
            max_results_per_miner=max_results_per_miner,
            max_response_time=max_response_time
        )

        if "results" in result:
            formatted_results = []
            for i, search_result in enumerate(result["results"], 1):
                url = search_result.get("url", "No URL")
                content = search_result.get("content", "No content")
                relevant = search_result.get("relevant", "No summary")

                formatted_results.append(f"""
**Result {i}:**
- **URL:** {url}
- **Content Preview:** {content[:200]}...
- **Relevant Info:** {relevant[:300]}...
""")

            return f"**Search Query:** {search_query}\n**Results from {n_miners} miners:**\n" + "\n".join(formatted_results)
        else:
            return f"Error: Unexpected response format: {result}"

    except Exception as e:
        return f"Error: {str(e)}"

# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    import uvicorn

    port = int(os.getenv("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    main()