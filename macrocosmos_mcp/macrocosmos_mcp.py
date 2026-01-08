import json
import os
from typing import List, Optional
import logging
from mcp.server.fastmcp import FastMCP
import macrocosmos as mc

# Initialize FastMCP server
mcp = FastMCP("macrocosmos")


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("macrocosmos_mcp_server")


# Constants - Using MC_API for unified authentication
MC_API = os.getenv("MC_API")

if not MC_API:
    logger.warning("MC_API environment variable not set")


@mcp.tool(
    description="""
Fetch real-time social media data from X (Twitter) and Reddit through the Macrocosmos SN13 network.
IMPORTANT: This tool requires 'source' parameter to be either 'X' or 'REDDIT' (case-sensitive).
Parameters:
- source (str, REQUIRED): Data platform - must be 'X' or 'REDDIT'
- usernames (List[str], optional): Up to 5 usernames to monitor.
  * For X: '@' symbol is optional (e.g., ['elonmusk', '@spacex'] both work)
  * NOT available for Reddit
- keywords (List[str], optional): Up to 5 keywords/hashtags to search
  * For X: any keywords or hashtags (e.g., ['AI', 'crypto', '#bitcoin'])
  * For Reddit: subreddit names (e.g., ['r/astronomy', 'space']) or 'r/all' for all subreddits
- start_date (str, optional): Start date/datetime in YYYY-MM-DD or ISO format
  * Examples: '2024-04-01' or '2024-01-01T00:00:00Z'
  * Defaults to 24 hours ago from current time if not specified
- end_date (str, optional): End date/datetime in YYYY-MM-DD or ISO format
  * Examples: '2024-04-25' or '2024-06-03T23:59:59Z'
  * Defaults to current time if not specified
- limit (int, optional): Maximum number of results to return (range: 1-1000, default: 10)
- keyword_mode (str, optional): How to match keywords - 'any' (default) or 'all'
  * 'any': returns posts matching ANY of the keywords
  * 'all': returns posts matching ALL of the keywords
Default Behavior (when dates not specified):
The tool searches the last 24 hours (from current time back to 24 hours ago).
Usage Examples:
1. Get recent tweets from specific users:
   query_on_demand_data(source='X', usernames=['@elonmusk', '@spacex'], limit=20)
2. Search tweets by keywords in last 24 hours:
   query_on_demand_data(source='X', keywords=['AI', 'machine learning'], limit=30)
3. Monitor specific users AND filter by keywords:
   query_on_demand_data(source='X', usernames=['@nasa'], keywords=['space', 'mars'], limit=20)
4. Monitor Reddit subreddits:
   query_on_demand_data(source='REDDIT', keywords=['r/astronomy', 'space'], limit=50)
5. Search across all of Reddit with date range:
   query_on_demand_data(source='REDDIT', keywords=['r/all', 'space'],
                       start_date='2025-04-01', end_date='2025-04-02', limit=50)
6. Strict keyword matching (requires ALL keywords):
   query_on_demand_data(source='X', keywords=['AI', 'machine learning'], keyword_mode='all', limit=30)
7. Precise datetime range search:
   query_on_demand_data(source='X', keywords=['Bitcoin'],
                       start_date='2024-06-01T00:00:00Z',
                       end_date='2024-06-03T23:59:59Z', limit=100)
Returns:
JSON object containing:
- status: "success" or error information
- data: Array of posts/tweets with full content, user information, engagement metrics,
        timestamps, platform-specific metadata, and media attachments
- meta: Processing statistics (miners queried, response rates, items returned, etc.)
Platform-Specific Notes:
- X (Twitter): '@' symbol is optional for usernames
- Reddit: Does NOT support username filtering, only subreddit/keyword searches
- All timestamps returned in UTC format
"""
)
async def query_on_demand_data(
    source: str,
    usernames: Optional[List[str]] = None,
    keywords: Optional[List[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: int = 10,
    keyword_mode: str = "any",
) -> str:
    """
    Query data on demand from various sources.

    Args:
        source: Data source: X or REDDIT
        usernames: List of usernames to filter by (X only, not available for Reddit)
        keywords: List of keywords to search for (Reddit: use r/subreddit format)
        start_date: Start date/datetime in YYYY-MM-DD or ISO format (e.g. 2024-04-01 or 2024-01-01T00:00:00Z)
        end_date: End date/datetime in YYYY-MM-DD or ISO format
        limit: Maximum number of items to return (1-1000)
        keyword_mode: How to match keywords - 'any' or 'all'
    """
    client = mc.AsyncSn13Client(api_key=MC_API)

    response = await client.sn13.OnDemandData(
        source=source,  # X or REDDIT
        usernames=usernames if usernames else [],  # Optional, up to 5 users
        keywords=keywords if keywords else [],  # Optional, up to 5 keywords
        start_date=start_date,  # Defaults to 24h range if not specified
        end_date=end_date,  # Defaults to current time if not specified
        limit=limit,  # Optional, up to 1000 results
        keyword_mode=keyword_mode,  # 'any' or 'all'
    )

    if not response:
        return "Failed to fetch data. Please check your API key and parameters."

    # Convert response to dict if it's a Pydantic model or similar object
    if hasattr(response, "model_dump"):
        response_dict = response.model_dump()
    elif hasattr(response, "dict"):
        response_dict = response.dict()
    elif isinstance(response, dict):
        response_dict = response
    else:
        response_dict = {"data": str(response)}

    # Return as JSON string - MCP expects string return type
    return json.dumps(response_dict, indent=2, default=str)


@mcp.tool(
    description="""
Create a Gravity task for large-scale data collection from X (Twitter) or Reddit.
Use this for collecting large datasets over time (up to 7 days). For quick queries (up to 1000 results), use query_on_demand_data instead.

The task registers on the network within 20 minutes and collects data for 7 days.
You'll receive an email notification when the dataset is ready for download.

Parameters:
- tasks (List[dict], REQUIRED): List of task objects, each containing:
  * platform (str): 'x' or 'reddit'
  * topic (str): The hashtag/subreddit to monitor
    - For X: MUST start with '#' or '$' (e.g., '#ai', '$BTC') - plain keywords are rejected!
    - For Reddit: subreddit name (e.g., 'r/MachineLearning')
  * keyword (str, optional): Additional keyword filter within the topic
    - Filters posts to only those containing this keyword
    - Example: topic='#Bittensor', keyword='dTAO' -> only #Bittensor posts mentioning 'dTAO'
- name (str, optional): Name for the task (helps organize multiple tasks)
- email (str, optional): Email address for notification when dataset is ready
- redirect_url (str, optional): URL to redirect to from the email notification

Returns:
- gravity_task_id: Unique identifier to track and manage the task

Examples:
1. Basic collection:
   create_gravity_task(
       tasks=[{"platform": "x", "topic": "#ai"}],
       name="AI Tweets"
   )

2. With keyword filter:
   create_gravity_task(
       tasks=[{"platform": "x", "topic": "#Bittensor", "keyword": "dTAO"}],
       name="Bittensor dTAO mentions"
   )

3. Multiple platforms:
   create_gravity_task(
       tasks=[
           {"platform": "x", "topic": "#ai", "keyword": "LLM"},
           {"platform": "reddit", "topic": "r/MachineLearning"}
       ],
       name="AI Data Collection",
       email="user@example.com"
   )
"""
)
async def create_gravity_task(
    tasks: List[dict],
    name: Optional[str] = None,
    email: Optional[str] = None,
    redirect_url: Optional[str] = "https://app.macrocosmos.ai/",
) -> str:
    """
    Create a Gravity task for large-scale data collection.

    Args:
        tasks: List of task objects with 'platform' and 'topic' keys
        name: Optional name for the task
        email: Optional email for notifications
        redirect_url: Optional redirect URL for email notifications
    """
    client = mc.AsyncGravityClient(api_key=MC_API)

    # Build notification requests if email provided
    notification_requests = []
    if email:
        notification_requests.append(
            {
                "type": "email",
                "address": email,
                "redirect_url": redirect_url or "https://app.macrocosmos.ai/",
            }
        )

    response = await client.gravity.CreateGravityTask(
        gravity_tasks=tasks,
        name=name,
        notification_requests=notification_requests if notification_requests else None,
    )

    if not response:
        return json.dumps(
            {
                "error": "Failed to create gravity task. Please check your API key and parameters."
            }
        )

    # Convert response to dict
    if hasattr(response, "model_dump"):
        response_dict = response.model_dump()
    elif hasattr(response, "dict"):
        response_dict = response.dict()
    elif isinstance(response, dict):
        response_dict = response
    else:
        response_dict = {"gravity_task_id": str(response)}

    return json.dumps(response_dict, indent=2, default=str)


@mcp.tool(
    description="""
Get the status of a Gravity task and see how much data has been collected.

Parameters:
- gravity_task_id (str, REQUIRED): The ID of the gravity task to check
- include_crawlers (bool, default: True): Whether to include detailed crawler information
  Set to True to see records_collected and bytes_collected for each crawler

Returns:
- Task status (Running, Completed, Pending, etc.)
- Task name and start time
- List of crawler IDs (needed for build_dataset)
- When include_crawlers=True: records_collected, bytes_collected per crawler

Example:
get_gravity_task_status(gravity_task_id="multicrawler-9f518ae4-xxxx-xxxx-xxxx-8b73d7cd4c49")
"""
)
async def get_gravity_task_status(
    gravity_task_id: str, include_crawlers: bool = True
) -> str:
    """
    Get the status of a Gravity task.

    Args:
        gravity_task_id: The unique identifier of the gravity task
        include_crawlers: Whether to include detailed crawler information
    """
    client = mc.AsyncGravityClient(api_key=MC_API)

    response = await client.gravity.GetGravityTasks(
        gravity_task_id=gravity_task_id, include_crawlers=include_crawlers
    )

    if not response:
        return json.dumps(
            {"error": "Failed to get gravity task status. Please check the task ID."}
        )

    # Convert response to dict
    if hasattr(response, "model_dump"):
        response_dict = response.model_dump()
    elif hasattr(response, "dict"):
        response_dict = response.dict()
    elif isinstance(response, dict):
        response_dict = response
    else:
        response_dict = {"data": str(response)}

    return json.dumps(response_dict, indent=2, default=str)


@mcp.tool(
    description="""
Build a dataset from collected data before the 7-day task completion.
Use this when you have enough data and don't want to wait for the full collection period.

WARNING: Building a dataset will STOP the crawler and de-register it from the network.
The crawler will no longer collect new data after this operation.

Parameters:
- crawler_id (str, REQUIRED): The ID of the crawler to build dataset from
  (Get this from get_gravity_task_status response - look for 'crawler_ids' field)
- max_rows (int, default: 10000): Maximum number of rows to include in the dataset
- email (str, optional): Email address for notification when dataset is ready
- redirect_url (str, optional): URL to redirect to from the email notification

Returns:
- dataset_id: Unique identifier to track the dataset build
- Build status and progress information (10 steps total)

Example:
build_dataset(
    crawler_id="crawler-0-multicrawler-9f518ae4-xxxx",
    max_rows=10000,
    email="user@example.com"
)
"""
)
async def build_dataset(
    crawler_id: str,
    max_rows: int = 10000,
    email: Optional[str] = None,
    redirect_url: Optional[str] = "https://app.macrocosmos.ai/",
) -> str:
    """
    Build a dataset from a crawler's collected data.

    Args:
        crawler_id: The ID of the crawler to build dataset from
        max_rows: Maximum number of rows to include (default: 10000)
        email: Optional email for notifications
        redirect_url: Optional redirect URL for email notifications
    """
    client = mc.AsyncGravityClient(api_key=MC_API)

    # Build notification requests - always required by the API
    if email:
        notification_requests = [
            {
                "type": "email",
                "address": email,
                "redirect_url": redirect_url or "https://app.macrocosmos.ai/",
            }
        ]
    else:
        # API requires notification_requests, pass minimal object
        notification_requests = [{"type": "email"}]

    response = await client.gravity.BuildDataset(
        crawler_id=crawler_id,
        max_rows=max_rows,
        notification_requests=notification_requests,
    )

    if not response:
        return json.dumps(
            {"error": "Failed to build dataset. Please check the crawler ID."}
        )

    # Convert response to dict
    if hasattr(response, "model_dump"):
        response_dict = response.model_dump()
    elif hasattr(response, "dict"):
        response_dict = response.dict()
    elif isinstance(response, dict):
        response_dict = response
    else:
        response_dict = {"data": str(response)}

    return json.dumps(response_dict, indent=2, default=str)


@mcp.tool(
    description="""
Get the status of a dataset build and download links when ready.

Parameters:
- dataset_id (str, REQUIRED): The ID of the dataset to check

Returns:
- Build status (Running, Completed, etc.)
- Progress steps (10 total steps)
- When completed: Download URLs for Parquet files
- File metadata (size, row count, expiration date)

Example:
get_dataset_status(dataset_id="dataset-71e97cfa-xxxx-xxxx-xxxx-33cd91be9028")
"""
)
async def get_dataset_status(dataset_id: str) -> str:
    """
    Get the status of a dataset build.

    Args:
        dataset_id: The unique identifier of the dataset
    """
    client = mc.AsyncGravityClient(api_key=MC_API)

    response = await client.gravity.GetDataset(dataset_id=dataset_id)

    if not response:
        return json.dumps(
            {"error": "Failed to get dataset status. Please check the dataset ID."}
        )

    # Convert response to dict
    if hasattr(response, "model_dump"):
        response_dict = response.model_dump()
    elif hasattr(response, "dict"):
        response_dict = response.dict()
    elif isinstance(response, dict):
        response_dict = response
    else:
        response_dict = {"data": str(response)}

    return json.dumps(response_dict, indent=2, default=str)


@mcp.tool(
    description="""
Cancel a running Gravity task and stop data collection.

Parameters:
- gravity_task_id (str, REQUIRED): The ID of the gravity task to cancel

Returns:
- Success or error message

Example:
cancel_gravity_task(gravity_task_id="multicrawler-9f518ae4-xxxx-xxxx-xxxx-8b73d7cd4c49")
"""
)
async def cancel_gravity_task(gravity_task_id: str) -> str:
    """
    Cancel a running Gravity task.

    Args:
        gravity_task_id: The unique identifier of the gravity task to cancel
    """
    client = mc.AsyncGravityClient(api_key=MC_API)

    response = await client.gravity.CancelGravityTask(gravity_task_id=gravity_task_id)

    if not response:
        return json.dumps(
            {"error": "Failed to cancel gravity task. Please check the task ID."}
        )

    # Convert response to dict
    if hasattr(response, "model_dump"):
        response_dict = response.model_dump()
    elif hasattr(response, "dict"):
        response_dict = response.dict()
    elif isinstance(response, dict):
        response_dict = response
    else:
        response_dict = {"message": str(response)}

    return json.dumps(response_dict, indent=2, default=str)


@mcp.tool(
    description="""
Cancel a dataset build or purge a completed dataset.

Parameters:
- dataset_id (str, REQUIRED): The ID of the dataset to cancel/purge

Returns:
- Success or error message

Example:
cancel_dataset(dataset_id="dataset-71e97cfa-xxxx-xxxx-xxxx-33cd91be9028")
"""
)
async def cancel_dataset(dataset_id: str) -> str:
    """
    Cancel a dataset build or purge a completed dataset.

    Args:
        dataset_id: The unique identifier of the dataset to cancel
    """
    client = mc.AsyncGravityClient(api_key=MC_API)

    response = await client.gravity.CancelDataset(dataset_id=dataset_id)

    if not response:
        return json.dumps(
            {"error": "Failed to cancel dataset. Please check the dataset ID."}
        )

    # Convert response to dict
    if hasattr(response, "model_dump"):
        response_dict = response.model_dump()
    elif hasattr(response, "dict"):
        response_dict = response.dict()
    elif isinstance(response, dict):
        response_dict = response
    else:
        response_dict = {"message": str(response)}

    return json.dumps(response_dict, indent=2, default=str)


def get_mcp():
    """Return the singleton FastMCP instance so other modules can re-use it."""
    return mcp


def main():
    """Main entry point for the MCP server."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
