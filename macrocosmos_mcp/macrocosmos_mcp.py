import os
from typing import Any, List, Optional, Dict
import logging
from mcp.server.fastmcp import FastMCP
import macrocosmos as mc

# Initialize FastMCP server
mcp = FastMCP("macrocosmos")


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("macrocosmos_mcp_server")


# Constants - Using MC_API for unified authentication
MC_API = os.getenv("MC_API")

if not MC_API:
    logger.warning("MC_API environment variable not set")


@mcp.tool(description="""
Fetch real-time social media data from X (Twitter), Reddit, and YouTube through the Macrocosmos SN13 network.
IMPORTANT: This tool requires ‘source’ parameter to be either ‘X’, ‘REDDIT’, or ‘YouTube’ (case-sensitive).
Parameters:
- source (str, REQUIRED): Data platform - must be ‘X’, ‘REDDIT’, or ‘YouTube’
- usernames (List[str], optional): Up to 5 usernames to monitor.
  * For X: include ‘@’ symbol (e.g., [‘@elonmusk’, ‘@spacex’])
  * For YouTube: channel names (e.g., [‘mrbeast’, ‘mkbhd’])
  * NOT available for Reddit
- keywords (List[str], optional): Up to 5 keywords/hashtags to search
  * For X: any keywords or hashtags (e.g., [‘AI’, ‘crypto’, ‘#bitcoin’])
  * For Reddit: subreddit names (e.g., [‘r/astronomy’, ‘space’]) or ‘r/all’ for all subreddits
  * For YouTube: search terms (e.g., [‘tutorial’, ‘review’])
- start_date (str, optional): Start date/datetime in YYYY-MM-DD or ISO format
  * Examples: ‘2024-04-01’ or ‘2024-01-01T00:00:00Z’
  * Defaults to 24 hours ago from current time if not specified
- end_date (str, optional): End date/datetime in YYYY-MM-DD or ISO format
  * Examples: ‘2024-04-25’ or ‘2024-06-03T23:59:59Z’
  * Defaults to current time if not specified
- limit (int, optional): Maximum number of results to return (range: 1-1000, default: 10)
- keyword_mode (str, optional): How to match keywords - ‘any’ (default) or ‘all’
  * ‘any’: returns posts matching ANY of the keywords
  * ‘all’: returns posts matching ALL of the keywords
Default Behavior (when dates not specified):
The tool searches the last 24 hours (from current time back to 24 hours ago).
Usage Examples:
1. Get recent tweets from specific users:
   query_on_demand_data(source=‘X’, usernames=[‘@elonmusk’, ‘@spacex’], limit=20)
2. Search tweets by keywords in last 24 hours:
   query_on_demand_data(source=‘X’, keywords=[‘AI’, ‘machine learning’], limit=30)
3. Monitor specific users AND filter by keywords:
   query_on_demand_data(source=‘X’, usernames=[‘@nasa’], keywords=[‘space’, ‘mars’], limit=20)
4. Search YouTube videos from specific channels:
   query_on_demand_data(source=‘YouTube’, usernames=[‘mrbeast’], start_date=‘2024-08-01’, limit=10)
5. Monitor Reddit subreddits:
   query_on_demand_data(source=‘REDDIT’, keywords=[‘r/astronomy’, ‘space’], limit=50)
6. Search across all of Reddit with date range:
   query_on_demand_data(source=‘REDDIT’, keywords=[‘r/all’, ‘space’],
                       start_date=‘2025-04-01’, end_date=‘2025-04-02’, limit=50)
7. Strict keyword matching (requires ALL keywords):
   query_on_demand_data(source=‘X’, keywords=[‘AI’, ‘machine learning’], keyword_mode=‘all’, limit=30)
8. Precise datetime range search:
   query_on_demand_data(source=‘X’, keywords=[‘Bitcoin’],
                       start_date=‘2024-06-01T00:00:00Z’,
                       end_date=‘2024-06-03T23:59:59Z’, limit=100)
Returns:
JSON object containing:
- status: “success” or error information
- data: Array of posts/tweets/videos with full content, user information, engagement metrics,
        timestamps, platform-specific metadata, and media attachments
- meta: Processing statistics (miners queried, response rates, items returned, etc.)
Platform-Specific Notes:
- X (Twitter): Usernames MUST include ‘@’ symbol
- Reddit: Does NOT support username filtering, only subreddit/keyword searches
- YouTube: Searches by channel username or keywords
- All timestamps returned in UTC format
""")
async def query_on_demand_data(
    source: str,
    usernames: Optional[List[str]] = None,
    keywords: Optional[List[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: int = 10,
    keyword_mode: str = "any"
) -> str:
    """
    Query data on demand from various sources.

    Args:
        source: Data source: X, REDDIT, or YouTube
        usernames: List of usernames to filter by (X: @username, YouTube: channel name, not available for Reddit)
        keywords: List of keywords to search for (Reddit: use r/subreddit format)
        start_date: Start date/datetime in YYYY-MM-DD or ISO format (e.g. 2024-04-01 or 2024-01-01T00:00:00Z)
        end_date: End date/datetime in YYYY-MM-DD or ISO format
        limit: Maximum number of items to return (1-1000)
        keyword_mode: How to match keywords - 'any' or 'all'
    """
    client = mc.AsyncSn13Client(api_key=MC_API)

    response = await client.sn13.OnDemandData(
        source=source,  # X, REDDIT, or YouTube
        usernames=usernames if usernames else [],  # Optional, up to 5 users
        keywords=keywords if keywords else [],  # Optional, up to 5 keywords
        start_date=start_date,  # Defaults to 24h range if not specified
        end_date=end_date,  # Defaults to current time if not specified
        limit=limit,  # Optional, up to 1000 results
        keyword_mode=keyword_mode  # 'any' or 'all'
    )

    if not response:
        return "Failed to fetch data. Please check your API key and parameters."

    return response


def get_mcp():
    """Return the singleton FastMCP instance so other modules can re-use it."""
    return mcp

def main():
    """Main entry point for the MCP server."""
    mcp.run(transport="stdio")
            
if __name__ == "__main__":
    main()