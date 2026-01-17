# Beads MCP Integration Instructions

I have successfully installed the `bd` CLI version 0.47.2 and initialized it in this workspace.
To fully integrate Beads with your agent workflow, you should configure the MCP server.

## 1. Install the MCP Server
Since you have Python 3.12, run:

```bash
pip install beads-mcp
```

## 2. Configure MCP
Add the following to your agent's MCP configuration file (typically `claude_desktop_config.json` or similar for your environment):

```json
{
  "mcpServers": {
    "beads": {
      "command": "uv",
      "args": [
        "tool",
        "run",
        "beads-mcp"
      ]
    }
  }
}
```

*Note: If you don't use `uv`, you can use `python -m beads_mcp` or simply `beads-mcp` if it's in your PATH.*

```json
{
  "mcpServers": {
    "beads": {
      "command": "beads-mcp"
    }
  }
}
```

## 3. Git Integration
Beads is designed to work with Git. This workspace is currently **not a git repository**.
To enable the full power of Beads (auto-sync, distributed tracking):

```bash
git init
git add .
git commit -m "Initial commit"
```

Once git is initialized, Beads will automatically sync your tasks to `.beads/issues.jsonl`.
