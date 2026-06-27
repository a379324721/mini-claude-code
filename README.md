# mini-claude-code

一个从零开始用 Python 实现的极简版 Claude Code。

## 快速开始

```bash
# 安装(需要 Python 3.11+)
pip install -e .

# 设置 API Key
export ANTHROPIC_API_KEY=sk-ant-...

# 运行
mini-claude-code "hello"                  # 一次性模式
mini-claude-code                          # 交互式 REPL
mini-claude-code --yolo "list files"      # 跳过确认
mini-claude-code --plan "refactor this"   # 计划模式
python -m mini_claude_code "hello"        # 也可以用 python -m 方式运行

# 使用 OpenAI 兼容后端
OPENAI_API_KEY=sk-xxx mini-claude-code --api-base https://api.openai.com/v1 --model gpt-4o "hello"
```

## 文件结构

```
mini-claude-code/
├── pyproject.toml
└── src/
    └── mini_claude_code/
        ├── __init__.py
        ├── __main__.py
        ├── agent.py
        ├── tools.py
        ├── prompt.py
        ├── ui.py
        ├── session.py
        ├── memory.py
        ├── skills.py
        ├── subagent.py
        ├── mcp_client.py
        └── frontmatter.py
```

| 文件 | 说明 |
|------|------|
| `agent.py` | Agent 核心循环、双后端、4 层压缩 |
| `tools.py` | 10 个工具 + 5 种权限模式 |
| `__main__.py` | CLI 入口与 REPL |
| `ui.py` | 终端 UI(rich) |
| `prompt.py` | 系统提示词构造 |
| `session.py` | 会话管理 |
| `memory.py` | 记忆系统 |
| `skills.py` | 技能系统 |
| `subagent.py` | 子 Agent |
| `mcp_client.py` | MCP 客户端 |
| `frontmatter.py` | YAML frontmatter 解析 |

## 依赖

- `anthropic` — Anthropic SDK(流式)
- `openai` — OpenAI SDK(兼容后端)
- `rich` — 终端彩色输出
