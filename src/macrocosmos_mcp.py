import os
from typing import Any, List, Optional, Dict
import httpx
import logging
from mcp.server.fastmcp import FastMCP
from mcp.types import Resource, TextContent
import time
import json
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
APEX_BASE_URL = "https://constellation.api.cloud.macrocosmos.ai"

if not MC_API:
    logger.warning("MC_API environment variable not set")


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
    client = mc.AsyncSn13Client(api_key=MC_API)

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


class ApexClient:
    """Client for interacting with the Apex API"""

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
        """Send a chat completion request to Apex"""

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
            try:
                response = await client.post(
                    f"{self.base_url}/apex.v1.ApexService/ChatCompletion",
                    headers=self.headers,
                    json=payload
                )
                response.raise_for_status()
                return response.json()
            except httpx.HTTPError as e:
                logger.error(f"HTTP error occurred: {e}")
                raise
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                raise

    async def web_search(
            self,
            search_query: str,
            n_miners: int = 3,
            max_results_per_miner: int = 2,
            max_response_time: int = 30
    ) -> Dict[str, Any]:
        """Perform web search using Apex's integrated retriever"""

        payload = {
            "search_query": search_query,
            "n_miners": n_miners,
            "max_results_per_miner": max_results_per_miner,
            "max_response_time": max_response_time
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                response = await client.post(
                    f"{self.base_url}/apex.v1.ApexService/WebRetrieval",
                    headers=self.headers,
                    json=payload
                )
                response.raise_for_status()
                return response.json()
            except httpx.HTTPError as e:
                logger.error(f"HTTP error occurred: {e}")
                raise
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                raise


@mcp.tool()
async def apex_chat(
        prompt: str,
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_new_tokens: int = 256,
        do_sample: bool = True,
        model: str = "Default"
) -> str:
    """
    Send a chat completion request to Apex (SN1) decentralized LLMs.

    Args:
        prompt: The prompt/question to send to the Apex model
        temperature: Controls randomness (0.0-1.0, default: 0.7)
        top_p: Controls nucleus sampling (0.0-1.0, default: 0.95)
        max_new_tokens: Maximum tokens to generate (default: 256)
        do_sample: Whether to use sampling (default: True)
        model: Model to use (default: "Default")

    Returns:
        The response from the Apex model
    """
    if not MC_API:
        return "Error: MC_API environment variable not set"

    try:
        client = ApexClient(MC_API)
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
async def apex_conversation(
        messages: str,
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_new_tokens: int = 512,
        do_sample: bool = True,
        model: str = "Default"
) -> str:
    """
    Send a multi-turn conversation to Apex models.

    Args:
        messages: JSON string of conversation messages in format:
                 [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]
        temperature: Controls randomness (0.0-1.0, default: 0.7)
        top_p: Controls nucleus sampling (0.0-1.0, default: 0.95)
        max_new_tokens: Maximum tokens to generate (default: 512)
        do_sample: Whether to use sampling (default: True)
        model: Model to use (default: "Default")

    Returns:
        The response from the Apex model
    """
    if not MC_API:
        return "Error: MC_API environment variable not set"

    try:
        # Parse the messages JSON
        conversation_messages = json.loads(messages)

        # Validate message format
        for msg in conversation_messages:
            if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
                return "Error: Invalid message format. Each message must have 'role' and 'content' fields."
            if msg["role"] not in ["user", "assistant", "system"]:
                return "Error: Invalid role. Must be 'user', 'assistant', or 'system'."

        client = ApexClient(MC_API)

        result = await client.chat_completion(
            messages=conversation_messages,
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

    except json.JSONDecodeError:
        return "Error: Invalid JSON format for messages parameter"
    except Exception as e:
        return f"Error: {str(e)}"


@mcp.tool()
async def apex_web_search(
        search_query: str,
        n_miners: int = 3,
        max_results_per_miner: int = 2,
        max_response_time: int = 30
) -> str:
    """
    Perform web search using Apex's decentralized web retrieval.

    Args:
        search_query: The search term or natural language query
        n_miners: Number of miners to use for search (default: 3)
        max_results_per_miner: Max results per miner (default: 2)
        max_response_time: Max wait time in seconds (default: 30)

    Returns:
        Formatted search results from multiple miners
    """
    if not MC_API:
        return "Error: MC_API environment variable not set"

    try:
        client = ApexClient(MC_API)

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

            return f"**Search Query:** {search_query}\n**Results from {n_miners} miners:**\n" + "\n".join(
                formatted_results)
        else:
            return f"Error: Unexpected response format: {result}"

    except Exception as e:
        return f"Error: {str(e)}"


@mcp.tool()
async def apex_performance_test(
        test_prompt: str,
        temperature: float = 0.7,
        iterations: int = 3
) -> str:
    """
    Test Apex model performance with multiple iterations of the same prompt.

    Args:
        test_prompt: The prompt to test with
        temperature: Controls randomness (0.0-1.0, default: 0.7)
        iterations: Number of times to run the test (default: 3, max: 5)

    Returns:
        Performance analysis with response times and consistency
    """
    if not MC_API:
        return "Error: MC_API environment variable not set"

    # Limit iterations to prevent abuse
    iterations = min(iterations, 5)

    try:
        client = ApexClient(MC_API)
        results = []

        for i in range(iterations):
            start_time = time.time()

            messages = [{"role": "user", "content": test_prompt}]
            result = await client.chat_completion(
                messages=messages,
                temperature=temperature,
                max_new_tokens=256
            )

            end_time = time.time()
            response_time = end_time - start_time

            if "choices" in result and len(result["choices"]) > 0:
                response_content = result["choices"][0]["message"]["content"]
                model_used = result.get("model", "Unknown")

                results.append({
                    "iteration": i + 1,
                    "response_time": round(response_time, 2),
                    "model": model_used,
                    "response": response_content,
                    "response_length": len(response_content)
                })
            else:
                results.append({
                    "iteration": i + 1,
                    "response_time": round(response_time, 2),
                    "error": "Invalid response format"
                })

        # Calculate performance metrics
        response_times = [r["response_time"] for r in results if "error" not in r]
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0

        # Format results
        formatted_output = [f"**Performance Test Results for:** {test_prompt[:50]}...\n"]
        formatted_output.append(f"**Average Response Time:** {avg_response_time:.2f} seconds\n")

        for result in results:
            if "error" in result:
                formatted_output.append(f"**Iteration {result['iteration']}:** Error - {result['error']}")
            else:
                formatted_output.append(f"""
**Iteration {result['iteration']}:**
- **Response Time:** {result['response_time']} seconds
- **Model:** {result['model']}
- **Response Length:** {result['response_length']} characters
- **Response:** {result['response'][:100]}...
""")

        return "\n".join(formatted_output)

    except Exception as e:
        return f"Error: {str(e)}"


@mcp.resource("apex://models")
async def list_apex_models() -> Resource:
    """List available information about Apex models and capabilities"""
    content = """
# Apex (SN1) Models and Capabilities

## About Apex
Apex is a decentralized agentic inference engine powered by Subnet 1 on the Bittensor network. It provides access to various open-source language models through a decentralized network of miners.

## Available Features
1. **Chat Completions** - Interactive conversations with decentralized LLMs
2. **Web Retrieval** - Web search with multiple miner consensus
3. **Deep Research** - Advanced multi-step research capabilities (advanced feature)

## Supported Models
- Default model selection (automatically chooses best available)
- Mistral models (e.g., mistral-small-3.1-24b-instruct)
- LLaMA variants
- Other open-source models depending on network availability

## Parameters
- **Temperature**: 0.0-1.0 (controls randomness)
- **Top-p**: 0.0-1.0 (nucleus sampling)
- **Max tokens**: Up to several thousand tokens
- **Sampling**: Enable/disable sampling

## Performance Testing
Use the `apex_performance_test` tool to compare response times and consistency across multiple requests.
"""

    return Resource(
        uri="apex://models",
        name="Apex Models Information",
        description="Information about available Apex models and capabilities",
        mimeType="text/markdown",
        text=content
    )


@mcp.prompt()
def compare_with_apex(question: str) -> list:
    """
    Generate a prompt to compare Claude's response with Apex models.

    Args:
        question: The question to test both models with
    """
    from mcp.server.fastmcp.prompts import base

    return [
        base.UserMessage(
            f"""I want to compare my response with decentralized LLMs on Apex. 

Please:
1. First, provide your own response to this question: "{question}"
2. Then use the apex_chat tool to get Apex's response to the same question
3. Finally, compare the two responses in terms of:
   - Accuracy and factual content
   - Creativity and style
   - Completeness
   - Response quality

Question to test: {question}"""
        )
    ]


def get_mcp():
    """Return the singleton FastMCP instance so other modules can re-use it."""
    return mcp



if __name__ == "__main__":
    mcp.run(transport="stdio")
