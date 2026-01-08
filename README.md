
# Macrocosmos MCP

<p align="center">
  Official Macrocosmos <a href="https://github.com/modelcontextprotocol">Model Context Protocol (MCP)</a> server that enables interaction with X (Twitter) and Reddit, powered by Data Universe (SN13) on Bittensor. This server allows MCP clients like <a href="https://www.anthropic.com/claude">Claude Desktop</a>, <a href="https://www.cursor.so">Cursor</a>, <a href="https://codeium.com/windsurf">Windsurf</a>, <a href="https://github.com/openai/openai-agents-python">OpenAI Agents</a> and others to fetch real-time social media data.
</p>

---

## Quickstart with Claude Desktop

1. Get your API key from [Macrocosmos](https://app.macrocosmos.ai/account?tab=api-keys). There is a free tier with $5 of credits to start.
2. Install `uv` (Python package manager), install with `curl -LsSf https://astral.sh/uv/install.sh | sh` or see the `uv` [repo](https://github.com/astral-sh/uv) for additional install methods.
3. Go to Claude > Settings > Developer > Edit Config > claude_desktop_config.json to include the following:

```json
{
  "mcpServers": {
    "macrocosmos": {
      "command": "uvx",
      "args": ["macrocosmos-mcp"],
      "env": {
        "MC_API": "<insert-your-api-key-here>"
      }
    }
  }
}
```

---

## Available Tools

### 1. `query_on_demand_data` - Real-time Social Media Queries

Fetch real-time data from X (Twitter) and Reddit. Best for quick queries up to 1000 results.

**Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `source` | string | **REQUIRED**. Platform: `'X'` or `'REDDIT'` (case-sensitive) |
| `usernames` | list | Up to 5 usernames. For X: `@` is optional. Not available for Reddit |
| `keywords` | list | Up to 5 keywords. For Reddit: first item is subreddit (e.g., `'r/MachineLearning'`) |
| `start_date` | string | ISO format (e.g., `'2024-01-01T00:00:00Z'`). Defaults to 24h ago |
| `end_date` | string | ISO format. Defaults to now |
| `limit` | int | Max results 1-1000. Default: 10 |
| `keyword_mode` | string | `'any'` (default) or `'all'` |

**Example prompts:**
- "What has @elonmusk been posting about today?"
- "Get me the latest posts from r/bittensor about dTAO"
- "Fetch 50 tweets about #AI from the last week"

---

### 2. `create_gravity_task` - Large-Scale Data Collection

Create a Gravity task for collecting large datasets over 7 days. Use this when you need more than 1000 results.

**Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `tasks` | list | **REQUIRED**. List of task objects (see below) |
| `name` | string | Optional name for the task |
| `email` | string | Email for notification when complete |

**Task object structure:**
```json
{
  "platform": "x",           // 'x' or 'reddit'
  "topic": "#Bittensor",     // For X: MUST start with '#' or '$'
  "keyword": "dTAO"          // Optional: filter within topic
}
```

**Important:** For X (Twitter), topics MUST start with `#` or `$` (e.g., `#ai`, `$BTC`). Plain keywords are rejected.

**Example prompts:**
- "Create a gravity task to collect #Bittensor tweets for the next 7 days"
- "Start collecting data from r/MachineLearning about neural networks"

---

### 3. `get_gravity_task_status` - Check Collection Progress

Monitor your Gravity task and see how much data has been collected.

**Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `gravity_task_id` | string | **REQUIRED**. The task ID from create_gravity_task |
| `include_crawlers` | bool | Include detailed stats. Default: `True` |

**Returns:** Task status, crawler IDs, records_collected, bytes_collected

**Example prompts:**
- "Check the status of my Bittensor data collection task"
- "How many records have been collected so far?"

---

### 4. `build_dataset` - Build & Download Dataset

Build a dataset from collected data before the 7-day completion.

**Warning:** This will STOP the crawler and de-register it from the network.

**Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `crawler_id` | string | **REQUIRED**. Get from get_gravity_task_status |
| `max_rows` | int | Max rows to include. Default: 10000 |
| `email` | string | Email for notification when ready |

**Example prompts:**
- "Build a dataset from my Bittensor crawler with 5000 rows"
- "I have enough data, build the dataset now"

---

### 5. `get_dataset_status` - Check Build Progress & Download

Check dataset build progress and get download links when ready.

**Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `dataset_id` | string | **REQUIRED**. The dataset ID from build_dataset |

**Returns:** Build status (10 steps), and when complete: download URLs for Parquet files

**Example prompts:**
- "Is my dataset ready to download?"
- "Get the download link for my Bittensor dataset"

---

### 6. `cancel_gravity_task` - Stop Data Collection

Cancel a running Gravity task.

**Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `gravity_task_id` | string | **REQUIRED**. The task ID to cancel |

---

### 7. `cancel_dataset` - Cancel Build or Purge Dataset

Cancel a dataset build or purge a completed dataset.

**Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `dataset_id` | string | **REQUIRED**. The dataset ID to cancel/purge |

---

## Example Workflows

### Quick Query (On-Demand)
```
User: "What's the sentiment about $TAO on Twitter today?"
→ Uses query_on_demand_data to fetch recent tweets
→ Returns up to 1000 results instantly
```

### Large Dataset Collection (Gravity)
```
User: "I need to collect a week's worth of #AI tweets for analysis"

1. create_gravity_task → Returns gravity_task_id
2. get_gravity_task_status → Monitor progress, get crawler_ids
3. build_dataset → When ready, build the dataset
4. get_dataset_status → Get download URL for Parquet file
```

---

## Example Prompts

### On-Demand Queries
- "What has the president of the U.S. been saying over the past week on X?"
- "Fetch me information about what people are posting on r/politics today."
- "Please analyze posts from @elonmusk for the last week."
- "Get me 100 tweets about #Bittensor and analyze the sentiment"

### Large-Scale Collection
- "Create a gravity task to collect data about #AI from Twitter and r/MachineLearning from Reddit"
- "Start a 7-day collection of $BTC tweets with keyword 'ETF'"
- "Check how many records my gravity task has collected"
- "Build a dataset with 10,000 rows from my crawler"

---

MIT License
Made with love by the Macrocosmos team
