import os
from typing import Any, List, Optional
import httpx
import logging
from mcp.server.fastmcp import FastMCP
import time
import macrocosmos as mc

# Initialize FastMCP server
mcp = FastMCP("macrocosmos")


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("macrocosmos_mcp_server")


# Constants
MC_KEY = os.getenv("SN1_API_KEY")


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
    client = mc.Sn13Client(api_key=MC_KEY)

    response = client.sn13.OnDemandData(
        source='X',  # or 'Reddit'
        usernames=usernames if usernames else [],  # Optional, up to 5 users
        keywords=keywords if keywords else [],  # Optional, up to 5 keywords
        start_date=start_date,  # Defaults to 24h range if not specified
        end_date=end_date,  # Defaults to current time if not specified
        limit=limit  # Optional, up to 1000 results
    )

    if not response:
        return "Failed to fetch data. Please check your API key and parameters."

    status = response.get("status")

    if status == "error":
        error_msg = result.get("meta", {}).get("error", "Unknown error")
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


if __name__ == "__main__":
    mcp.run(transport='stdio')
