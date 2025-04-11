import os
from typing import Any, List, Optional
import httpx
import logging
from mcp.server.fastmcp import FastMCP
import time


# Initialize FastMCP server
mcp = FastMCP("macrocosmos")


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("macrocosmos_mcp_server")


# Constants
SN13_API_BASE = "https://sn13.api.macrocosmos.ai/api/v1"
SN1_API_BASE = "https://sn1.api.macrocosmos.ai"
SN13_API_KEY = os.getenv("SN13_API_KEY")
SN1_API_KEY = os.getenv("SN1_API_KEY")


@mcp.tool(description='Tool to fetch the data from X and Reddit, the data-source should X or Reddit!!!!')
async def query_on_demand_data(
    source: str,
    usernames: Optional[List[str]] = None,
    keywords: Optional[List[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: int = 10
) -> str:
    """
    Query data on demand from various sources.

    Args:
        source: Data source:  X, REDDIT
        usernames: List of usernames to filter by
        keywords: List of keywords to search for
        start_date: Start date in ISO format (e.g. 2023-01-01T00:00:00Z)
        end_date: End date in ISO format
        limit: Maximum number of items to return
    """
    headers = {
        "Content-Type": "application/json",
        "X-API-Key": SN13_API_KEY
    }

    request_data = {
        "source": source,
        "usernames": usernames or [],
        "keywords": keywords or [],
        "start_date": start_date,
        "end_date": end_date,
        "limit": limit
    }

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(f"{SN13_API_BASE}/on_demand_data_request",
                                     headers=headers, json=request_data, timeout=30.0)
            response.raise_for_status()
            result = response.json()
        except Exception as e:
            logger.error(f"API request error: {e}")
            return e

    if not result:
        return "Failed to fetch data. Please check your API key and parameters."

    status = result.get("status")

    if status == "error":
        error_msg = result.get("meta", {}).get("error", "Unknown error")
        return f"Error: {error_msg}"

    data = result.get("data", [])
    meta = result.get("meta", {})

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
            UserInfo: {item.get('user', 'N/A'),}
            Media: {item.get('media'), 'No media'}
        """
        formatted_data.append(formatted_item)

    meta_info = f"""
    Meta Information:
        - Miners queried: {meta.get('miners_queried', 'N/A')}
        - Data source: {meta.get('source', 'N/A')}
        - Items returned: {meta.get('items_returned', len(data))}
    """

    return meta_info + "\n\n" + "\n---\n".join(formatted_data)


@mcp.tool(description='Tool to get HF repository and dataset list uploaded by SN13')
async def list_hf_repos() -> str:
    """
    Get a list of Hugging Face repository names using a GET request.
    """
    headers = {
        "Content-Type": "application/json",
        "X-API-Key": SN13_API_KEY
    }

    url = f"{SN13_API_BASE.rstrip('/')}/list_repo_names"
    logging.info(f"Fetching Hugging Face repositories from {url}")
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, headers=headers, timeout=30.0)
            response.raise_for_status()
            result = response.json()
        except Exception as e:
            logger.error(f"API request error: {e}")
            return e

    count = result.get("count", 0)
    repos = result.get("repo_names", [])

    if not repos:
        return "No repositories found."

    return f"Found {count} Hugging Face repositories:\n\n" + "\n".join(repos)


@mcp.tool(description="Tool to perform the web-search with apex")
async def benchmark_search_queries(
    search_queries: List[str],
    n_miners: int = 5,
    n_results: int = 5,
    max_response_time: int = 10,
    timeout_seconds: float = 12.0
) -> str:
    """
    Tool to perform the web-search with apex
    Benchmark search queries by measuring response time and result count.

    Args:
        search_queries: List of search strings to query.
        n_miners: Number of miners to query.
        n_results: Number of results per query.
        max_response_time: Maximum time miners can take to respond.
        timeout_seconds: Timeout for each API call.

    Returns:
        A summary string with average result length and average response time.
    """
    headers = {
        "accept": "application/json",
        "api-key": SN1_API_KEY,
        "Content-Type": "application/json",
    }

    total_length = 0
    total_time = 0.0
    query_count = 0

    async with httpx.AsyncClient() as client:
        for query in search_queries:
            payload = {
                "search_query": query,
                "n_miners": n_miners,
                "n_results": n_results,
                "max_response_time": max_response_time,
            }

            start_time = time.perf_counter()
            try:
                response = await client.post(f"{SN1_API_BASE}/web_retrieval", headers=headers, json=payload, timeout=timeout_seconds)
                response.raise_for_status()
                data = response.json()
                elapsed = time.perf_counter() - start_time
                results = data.get("results", [])
                result_length = len(results) if isinstance(results, list) else 0

                total_length += result_length
                total_time += elapsed
                query_count += 1

                print(f"{query} : {result_length} : {elapsed:.3f}s")

            except Exception as e:
                elapsed = time.perf_counter() - start_time
                return e
                print(f"Query failed: {query} ({elapsed:.3f}s). Error: {e}")

    if query_count == 0:
        return "No successful queries to compute stats."

    average_length = total_length / query_count
    average_time = total_time / query_count

    return (
        f"Benchmark completed on {query_count} queries.\n"
        f"- Average result length: {average_length:.2f}\n"
        f"- Average response time: {average_time:.3f} seconds"
    )

if __name__ == "__main__":
    mcp.run(transport='stdio')
