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
MC_KEY = os.getenv("MC_KEY")


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

Note: Reddit does not support username filtering - use subreddit names in the keywords parameter instead. All X/Twitter usernames must include the '@' symbol.
""")
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
    client = mc.AsyncSn13Client(api_key=MC_KEY)

    response = await client.sn13.OnDemandData(
        source=source,  # or 'Reddit'
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
