
# 🌌 Macrocosmos MCP

<p align="center">
  Official Macrocosmos <a href="https://github.com/modelcontextprotocol">Model Context Protocol (MCP)</a> server that enables interaction with X, Reddit and Youtube, powered by Data Universe (SN13) on Bittensor. This server allows MCP clients like <a href="https://www.anthropic.com/claude">Claude Desktop</a>, <a href="https://www.cursor.so">Cursor</a>, <a href="https://codeium.com/windsurf">Windsurf</a>, <a href="https://github.com/openai/openai-agents-python">OpenAI Agents</a> and others to fetch real-time social media and video transcript data.
</p>

---

## Quickstart with Claude Desktop

1. Get your API key from [Macrocosmos](https://app.macrocosmos.ai/account?tab=api-keys). There is a free tier with $5 of credits to start.
2. Install `uv` (Python package manager), install with `curl -LsSf https://astral.sh/uv/install.sh | sh` or see the `uv` [repo](https://github.com/astral-sh/uv) for additional install methods.
3. Go to Claude > Settings > Developer > Edit Config > claude_desktop_config.json to include the following:

```
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

## Example usage

⚠️ Warning: Macrocosmos credits are needed to use these tools.

Try asking Claude:

- "What has the president of the U.S. been saying over the past week on X?"
- "Fetch me information about what people are posting on r/politics today."
- "Please analyze posts from @elonmusk for the last week."
- "Can you summarize the transcript of the latest video from WolfeyVGC?"


## 🔮 Upcoming

- 🧠 All the power of **Subnets** in your AI environment — coming soon.

---

MIT License
Made with ❤️ by the Macrocosm OS team
