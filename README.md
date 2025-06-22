# 🌌 Macrocosmos MCP

**Macrocosmos MCP** lets you integrate **SN13** and **SN1** APIs directly into **Claude for Desktop** or **Cursor**. Instantly tap into social data, perform live web searches, and explore Hugging Face models — all from within your AI environment.

---

## ⚡ Features

- 🔍 Query **X** (Twitter) and **Reddit** data on demand
- 📚 Explore **SN13 Hugging Face** repositories and datasets

---

## 🚀 Quick Setup

### 1. Clone the Repo

```bash
git clone https://github.com/macrocosm-os/macrocosmos-mcp.git
cd macrocosmos-mcp/src
```

### 2. Install Requirements (requires [uv](https://astral.sh/blog/uv/))

```bash
uv venv
source .venv/bin/activate
uv add "mcp[cli]" httpx macrocosmos
```

### 3. Configure Claude or Cursor

Open the MCP config file:

- **Claude:** `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Cursor:** `~/Library/Application Support/Cursor/cursor_mcp_config.json`

Add this entry:

```json
"macrocosmos": {
    "command": "uv",
    "args": [
        "--directory",
        "/full_path_to_repo/macrocosmos_mcp/src/",
        "run",
        "macrocosmos_mcp.py"
    ],
    "env": {
        "MC_API": "",
    }
}
```

Replace `/full_path_to_repo/` with your full local path and insert your API keys.

---

## 🔮 Upcoming

- 🧠 All the power of **Subnets** in your AI environment — coming soon.

---

MIT License
Made with ❤️ by the Macrocosm OS team
