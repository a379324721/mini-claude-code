# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 运行 / 开发

项目用 `uv` 管理依赖,所有命令前缀 `uv run` 即可在虚拟环境里执行;无需手动激活。

```bash
uv run python -m mini_claude_code "hello"         # 一次性模式
uv run python -m mini_claude_code                 # 交互式 REPL
uv run python -m mini_claude_code --yolo "..."    # 跳过权限确认
uv run python -m mini_claude_code --plan "..."    # 计划模式(只读)
```

没有测试套件,也没有 lint 配置。改完代码要做端到端验证就跑一次 `uv run python -m mini_claude_code --yolo --max-turns 2 "用 list_files 列目录"`,确认流式文本、工具调用、token、cost 都正常。

## 高层架构

### Backend 抽象是核心分界

Agent 同时支持 Anthropic 原生 API 和 OpenAI 兼容 API(DeepSeek / Qwen / vLLM 等),两套 SDK 的消息形态、流式协议、压缩策略差异都封装在 `backends/`:

- `backends/base.py` — `Backend` ABC,定义统一接口:消息追加、`stream()`、4 层压缩、`side_query()`、`serialize/restore`、`child_config()`、`estimate_cost_usd()`
- `backends/anthropic.py` — Anthropic SDK 实现,消息是 `list[dict]`,user 内容可以是 str 或 content blocks(tool_result 嵌在 user 列表,tool_use 嵌在 assistant 列表)
- `backends/openai.py` — OpenAI SDK 实现,消息严格 `[system, user, assistant?, tool, ...]`,tool 调用通过 `assistant.tool_calls` + `role="tool"` + `tool_call_id` 闭环

**Agent 主循环不接触原始 SDK 对象**。`stream()` 内部 commit assistant 消息后返回归一化的 `(tool_uses: list[NormalizedToolUse], usage: dict)`,Agent 据此调度。回写 tool 结果用 `ToolResult` dataclass 批量 append。

工具定义内部统一用 Anthropic 风格(`input_schema`);OpenAI 后端在自己的 `_to_openai_tools` 里把 `input_schema` 包成 OpenAI 的 `function.parameters` 格式。新加工具时只写 Anthropic schema。

### 流式工具早启动

只读、无副作用的工具(`CONCURRENCY_SAFE_TOOLS` = `{read_file, list_files, grep_search, web_fetch}`)在流式期间一旦完整生成就立刻 `asyncio.create_task` 启动,放进主循环的 `early_executions` dict。模型还在生成文本时,工具就已经在跑了。

- Anthropic 后端用 `content_block_stop` 事件触发 `on_tool_block_complete` 回调
- OpenAI 后端无对应事件,主循环在 `stream()` 返回后做一次兜底 sweep 启动剩余的并发安全工具,等价 `asyncio.gather`

修改主循环时要保证:`stream()` 重试(retry)和 context_break 时,未消费的 early task 都要通过 `_cancel_pending_early()` 取消并主动 consume exception,避免孤儿。

### 4 层压缩流水线

每回合边界自动跑(`backend.run_compression_pipeline()`):

1. **`budget_tool_results`** — 利用率 ≥0.5 时,单条 tool result 超 15-30KB 中间裁剪
2. **`snip_stale_results`** — 利用率 ≥0.6 时,只保留最近 3 条 snippable tool 结果(`read_file/grep_search/list_files/run_shell`),其余替换为 `SNIP_PLACEHOLDER`
3. **`microcompact`** — 距上次 API 调用 ≥5 分钟,把超过 3 条之外的旧 tool result 清空为 `CLEARED_PLACEHOLDER`

第四层是 `compact_conversation()`——`last_input_token_count > 0.85 * window` 时调用模型生成摘要重建消息。该方法只能在最后一条是 user/assistant 普通文本时调,否则切片会把 tool_use ↔ tool_result 对截断;`_is_safe_compact_tail()` 在 re-append 末尾消息前做这个判断。

### Plan 模式 + clear-and-execute 流程

`enter_plan_mode` / `exit_plan_mode` 是工具(在 `agent.py` 内部分发),不通过 `tools.py`。`exit_plan_mode` 的审批回调由 `__main__.py` 的 `plan_approval_fn` 注入。

`clear-and-execute` 路径要求:工具返回前清空历史 + 把当前工具结果作为新 user 消息塞回去。机制是 Agent 设置 `_context_cleared = True`,主循环在 tool 结果处理时检测到该标志 → 调 `backend.append_user_after_context_clear()` → `context_break = True` 跳出 for 循环。修改时小心不要让 `_context_cleared` 漏 reset,否则会在下次工具调用时错误地再次清空。

### 子 Agent / Skill fork

`_execute_agent_tool` 和 `_execute_skill_tool` 构造子 Agent 时**必须**用 `**self.backend.child_config()` 透传完整后端配置(api_base + anthropic_base_url + api_key),否则在用户用自定义 endpoint 时会撞回默认 SDK URL。

子 Agent 的输出通过 `_output_buffer: list[str]` 捕获,而不是直接 print;`run_once()` 是入口。

### 记忆召回(非阻塞)

每个用户回合开始时 `start_memory_prefetch()` 启动后台 task(签名:`async (system, user) -> str`,从 `backend.side_query()` 取得)。主循环每次迭代检查 prefetch 是否 settled,settled 后通过 `append_user_text_inline_or_new()` 把召回结果追加到最后一条 user 消息——`AnthropicBackend` 要兼容 str 和 list 两种 content 形态(后者 push 一个 `{"type":"text"}` 块)。

### 会话持久化

`session.py` 写 JSON 到 `~/.mini-claude/sessions/`,schema 同时保留 `anthropicMessages` 和 `openaiMessages` 两个键(其中之一为 None)。这是为了向后兼容旧 session 文件——不要重命名或合并这两个键。`backend.restore(data)` 返回 bool,Agent 用它检测跨后端 restore 是否失败(saved 是 OpenAI session 但当前 backend 是 Anthropic 之类),给出明确提示。

### 工具懒加载 / MCP

`get_active_tool_definitions(all_tools)` 过滤 deferred 字段后才暴露给模型——某些 schema 庞大的工具(如 `agent` 的 type 列表)首次被引用前不计入 system prompt。

MCP 工具按 `mcp__<server>__<tool>` 命名前缀路由,由 `McpManager` 在 Agent 首次 `chat()` 时懒加载(`load_and_connect()`),拼到 `self.tools` 后面。

## 提交风格

每个 bug 修复一个独立 commit,commit message 用 `fix(scope): 简述` 中文标题 + 中文 body(说清触发场景和后果)。所有 commit 末尾带 `Co-Authored-By: Claude` 行(见现有 commit history)。
