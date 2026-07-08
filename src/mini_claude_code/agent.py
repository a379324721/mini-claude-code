"""Agent 核心循环 —— 通过 Backend 抽象屏蔽 Anthropic/OpenAI 差异。

保留功能: 流式工具早启动、4 层压缩、Plan 模式、子 Agent、Skill fork、MCP、
权限检查、记忆召回、大结果持久化、预算控制。
"""

from __future__ import annotations

import asyncio
import time
import uuid
from pathlib import Path
from typing import Awaitable, Callable

from .tools import (
    tool_definitions,
    execute_tool,
    check_permission,
    CONCURRENCY_SAFE_TOOLS,
    get_active_tool_definitions,
    ToolDef,
)
from .memory import (
    start_memory_prefetch,
    format_memories_for_injection,
    MemoryPrefetch,
)
from .ui import (
    print_tool_call,
    print_tool_result,
    confirm_dangerous,
    print_retry,
    print_info,
    print_sub_agent_start,
    print_sub_agent_end,
    OutputSink,
    ConsoleSink,
    BufferSink,
)
from .session import save_session
from .prompt import build_system_prompt
from .subagent import get_sub_agent_config, inject_agent_types
from .mcp_client import McpManager
from .backends import (
    Backend,
    NormalizedToolUse,
    ToolResult,
    make_backend,
    get_context_window,
)
from .backends.base import (
    model_supports_thinking,
    model_supports_adaptive_thinking,
    sanitize_surrogates,
)


# ─── 工具 ─────────────────────────────────────────────────────


def _cancel_pending_early(tasks: dict[str, asyncio.Task]) -> None:
    """取消尚未消费的早启动工具 task,并通过 done_callback 主动 consume
    exception,避免 asyncio 打 'Task exception was never retrieved'。"""
    for t in tasks.values():
        if not t.done():
            t.cancel()
        t.add_done_callback(lambda done: done.exception() if not done.cancelled() else None)
    tasks.clear()


# ─── Agent ───────────────────────────────────────────────────


class Agent:
    def __init__(
        self,
        *,
        permission_mode: str = "default",
        model: str = "claude-opus-4-6",
        api_base: str | None = None,
        anthropic_base_url: str | None = None,
        api_key: str | None = None,
        thinking: bool = False,
        max_cost_usd: float | None = None,
        max_turns: int | None = None,
        custom_system_prompt: str | None = None,
        custom_tools: list[ToolDef] | None = None,
        is_sub_agent: bool = False,
    ):
        self.permission_mode = permission_mode
        self.thinking = thinking
        self.model = model
        self.is_sub_agent = is_sub_agent
        self.tools = custom_tools or tool_definitions
        self.max_cost_usd = max_cost_usd
        self.max_turns = max_turns
        self.session_id = uuid.uuid4().hex[:8]
        self.session_start_time = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cache_read_tokens = 0
        self.total_cache_creation_tokens = 0
        self.current_turns = 0

        # 中断支持
        self._aborted = False
        self._current_task: asyncio.Task | None = None

        # 权限白名单
        self._confirmed_paths: set[str] = set()

        # Plan 模式状态
        self._pre_plan_mode: str | None = None
        self._plan_file_path: str | None = None
        self._plan_approval_fn: Callable[[str], Awaitable[dict]] | None = None
        # plan 约束文案待注入到下一条 user 消息(用于没有 tool_result 通道的
        # 路径: 手动 /plan 与初始 --plan)。绝不写进 system,以免破坏缓存前缀。
        self._pending_plan_notice: str | None = None
        self._context_cleared: bool = False  # plan 审批清空上下文时置位

        # Thinking 模式
        self._thinking_mode = self._resolve_thinking_mode()

        # 输出通道: 主 Agent 打终端,子 Agent 捕获到缓冲(run_once 取走)
        self.sink: OutputSink = BufferSink() if is_sub_agent else ConsoleSink()

        # 编辑前必须先读: 跟踪文件读取时间戳(absolutePath → mtime)
        self._read_file_state: dict[str, float] = {}

        # MCP 集成
        self._mcp_manager = McpManager()
        self._mcp_initialized = False

        # 记忆召回状态 —— 每次用户回合做一次语义预取
        self._already_surfaced_memories: set[str] = set()
        self._session_memory_bytes = 0

        # 系统提示词 —— 整个会话生命周期恒定(只含 base),plan 约束走 message
        # 通道传达。这样 system 缓存断点的前缀永不变化,进出 plan 模式不会
        # 使 tools+system 的缓存失效。
        self._base_system_prompt = custom_system_prompt or build_system_prompt()
        self._system_prompt = self._base_system_prompt
        if self.permission_mode == "plan":
            self._plan_file_path = self._generate_plan_file_path()
            self._pending_plan_notice = self._build_plan_mode_prompt()

        # 后端
        self.backend: Backend = make_backend(
            model=model,
            system_prompt=self._system_prompt,
            emit_text=self.sink.emit_text,
            stop_spinner=self.sink.spinner_stop,
            api_base=api_base,
            anthropic_base_url=anthropic_base_url,
            api_key=api_key,
            on_retry=lambda attempt, m, reason: print_retry(attempt, m, reason),
        )

    # ─── Thinking 模式解析 ─────────────────────────────

    def _resolve_thinking_mode(self) -> str:
        if not self.thinking:
            return "disabled"
        if not model_supports_thinking(self.model):
            return "disabled"
        if model_supports_adaptive_thinking(self.model):
            return "adaptive"
        return "enabled"

    @property
    def is_processing(self) -> bool:
        return self._current_task is not None and not self._current_task.done()

    def abort(self) -> None:
        self._aborted = True
        if self._current_task and not self._current_task.done():
            self._current_task.cancel()

    def set_plan_approval_fn(self, fn: Callable[[str], Awaitable[dict]]) -> None:
        self._plan_approval_fn = fn

    # ─── Plan 模式切换 ────────────────────────────────────

    def toggle_plan_mode(self) -> str:
        if self.permission_mode == "plan":
            self.permission_mode = self._pre_plan_mode or "default"
            self._pre_plan_mode = None
            self._plan_file_path = None
            self._pending_plan_notice = None
            print_info(f"已退出 plan 模式 → {self.permission_mode} 模式")
            return self.permission_mode
        else:
            self._pre_plan_mode = self.permission_mode
            self.permission_mode = "plan"
            self._plan_file_path = self._generate_plan_file_path()
            self._pending_plan_notice = self._build_plan_mode_prompt()
            print_info(f"已进入 plan 模式。Plan 文件: {self._plan_file_path}")
            return "plan"

    def get_token_usage(self) -> dict:
        return {"input": self.total_input_tokens, "output": self.total_output_tokens}

    # ─── 主入口 ────────────────────────────────────

    async def chat(self, user_message: str) -> None:
        # 首次对话时懒加载连接 MCP 服务器(仅主 Agent)
        if not self._mcp_initialized and not self.is_sub_agent:
            self._mcp_initialized = True
            try:
                await self._mcp_manager.load_and_connect()
                mcp_defs = self._mcp_manager.get_tool_definitions()
                if mcp_defs:
                    self.tools = self.tools + mcp_defs
            except Exception as e:
                print(f"[mcp] 初始化失败: {e}", flush=True)

        self._aborted = False
        self._current_task = asyncio.current_task()
        try:
            await self._chat(user_message)
        except asyncio.CancelledError:
            self._aborted = True
        finally:
            self._current_task = None
        self.sink.session_end()
        if not self.is_sub_agent:
            self._auto_save()

    # ─── 子 Agent 入口 ────────────────────────────────

    async def run_once(self, prompt: str) -> dict:
        prev_in = self.total_input_tokens
        prev_out = self.total_output_tokens
        prev_cache_read = self.total_cache_read_tokens
        prev_cache_creation = self.total_cache_creation_tokens
        await self.chat(prompt)
        # is_sub_agent=True 时 sink 必为 BufferSink,输出已被捕获在缓冲里
        text = self.sink.take()
        return {
            "text": text,
            "tokens": {
                "input": self.total_input_tokens - prev_in,
                "output": self.total_output_tokens - prev_out,
                "cache_read": self.total_cache_read_tokens - prev_cache_read,
                "cache_creation": self.total_cache_creation_tokens - prev_cache_creation,
            },
        }

    # ─── REPL 命令 ────────────────────────────────────────

    def clear_history(self) -> None:
        self.backend.clear_history()
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cache_read_tokens = 0
        self.total_cache_creation_tokens = 0
        print_info("会话已清空。")

    def show_cost(self) -> None:
        total = self._get_current_cost_usd()
        turn_info = f" | 回合: {self.current_turns}/{self.max_turns}" if self.max_turns else ""
        if total is None:
            print_info(
                f"Token: 输入 {self.total_input_tokens} / 输出 {self.total_output_tokens}\n"
                f"  (当前后端无内置价目表,不估算费用){turn_info}"
            )
            return
        budget_info = f" / 预算 ${self.max_cost_usd}" if self.max_cost_usd else ""
        cache_info = ""
        if self.total_cache_read_tokens or self.total_cache_creation_tokens:
            cache_info = (
                f"\n  缓存: 命中 {self.total_cache_read_tokens}"
                f" / 写入 {self.total_cache_creation_tokens}"
            )
        print_info(
            f"Token: 输入 {self.total_input_tokens} / 输出 {self.total_output_tokens}{cache_info}\n"
            f"  预估费用: ${total:.4f}{budget_info}{turn_info}"
        )

    def _get_current_cost_usd(self) -> float | None:
        return self.backend.estimate_cost_usd(
            self.total_input_tokens,
            self.total_output_tokens,
            self.total_cache_read_tokens,
            self.total_cache_creation_tokens,
        )

    def _check_budget(self) -> dict:
        cost = self._get_current_cost_usd()
        if self.max_cost_usd is not None and cost is not None and cost >= self.max_cost_usd:
            return {"exceeded": True, "reason": f"费用上限已达 (${cost:.4f} >= ${self.max_cost_usd})"}
        if self.max_turns is not None and self.current_turns >= self.max_turns:
            return {"exceeded": True, "reason": f"回合数上限已达 ({self.current_turns} >= {self.max_turns})"}
        return {"exceeded": False}

    async def compact(self) -> None:
        if await self.backend.compact_conversation():
            print_info("会话已压缩。")
        else:
            print_info("会话太短,无需压缩。")

    # ─── 会话 ──────────────────────────────────────────────

    def restore_session(self, data: dict) -> None:
        if self.backend.restore(data):
            print_info(f"会话已恢复(共 {self.backend.message_count()} 条消息)。")
            return
        # 当前后端没找到自己的消息 —— 多半是上次保存时用的另一个后端。
        other = "openaiMessages" if data.get("openaiMessages") else (
            "anthropicMessages" if data.get("anthropicMessages") else None
        )
        if other:
            print_info(
                f"会话恢复失败:saved session 含 {other},当前后端不兼容。"
                "请切换到匹配的后端再 --resume(或先 /clear)。"
            )
        else:
            print_info("会话恢复失败:session 数据为空。")

    def _auto_save(self) -> None:
        try:
            save_session(self.session_id, {
                "metadata": {
                    "id": self.session_id,
                    "model": self.model,
                    "cwd": str(Path.cwd()),
                    "startTime": self.session_start_time,
                    "messageCount": self.backend.message_count(),
                },
                **self.backend.serialize(),
            })
        except Exception:
            pass

    # ─── 自动压缩 ──────────────────────────────────────────

    async def _check_and_compact(self) -> None:
        bw = self.backend.effective_window
        if self.backend.last_input_token_count > bw * 0.85:
            print_info("上下文窗口即将填满,正在压缩会话……")
            if await self.backend.compact_conversation():
                print_info("会话已压缩。")

    # ─── 大结果持久化 ─────────────────────────────────
    # 当工具结果超过 30 KB,把它写入磁盘,并把上下文里那条记录替换成
    # 简短预览 + 文件路径。模型之后可以用 read_file 取回完整输出 ——
    # 信息不丢失。

    def _persist_large_result(self, tool_name: str, result: str) -> str:
        THRESHOLD = 30 * 1024  # 30 KB
        if len(result.encode()) <= THRESHOLD:
            return result
        d = Path.home() / ".mini-claude" / "tool-results"
        d.mkdir(parents=True, exist_ok=True)
        filename = f"{int(time.time() * 1000)}-{tool_name}.txt"
        filepath = d / filename
        filepath.write_text(result, encoding="utf-8")

        lines = result.split("\n")
        preview = "\n".join(lines[:200])
        size_kb = len(result.encode()) / 1024

        return (
            f"[结果过大({size_kb:.1f} KB, 共 {len(lines)} 行)。"
            f"完整输出已保存到 {filepath}。"
            f"如需查看完整结果可以使用 read_file。]\n\n"
            f"预览(前 200 行):\n{preview}"
        )

    # ─── 执行工具(内部分发 agent/skill/plan mode) ─────

    async def _execute_tool_call(self, name: str, inp: dict) -> str:
        if name in ("enter_plan_mode", "exit_plan_mode"):
            return await self._execute_plan_mode_tool(name)
        if name == "agent":
            return await self._execute_agent_tool(inp)
        if name == "skill":
            return await self._execute_skill_tool(inp)
        # 把 MCP 工具调用路由给 MCP manager
        if self._mcp_manager.is_mcp_tool(name):
            return await self._mcp_manager.call_tool(name, inp)
        return await execute_tool(name, inp, self._read_file_state)

    # ─── Skill fork 模式 ─────────────────────────────────────

    async def _execute_skill_tool(self, inp: dict) -> str:
        from .skills import execute_skill
        result = execute_skill(inp.get("skill_name", ""), inp.get("args", ""))
        if not result:
            return f"未知 skill: {inp.get('skill_name', '')}"

        if result["context"] == "fork":
            tools = (
                [t for t in self.tools if t["name"] in result["allowed_tools"]]
                if result.get("allowed_tools")
                else [t for t in self.tools if t["name"] != "agent"]
            )
            print_sub_agent_start("skill-fork", inp.get("skill_name", ""))
            sub_agent = Agent(
                model=self.model,
                custom_system_prompt=result["prompt"],
                custom_tools=tools,
                is_sub_agent=True,
                permission_mode="plan" if self.permission_mode == "plan" else "bypassPermissions",
                **self.backend.child_config(),
            )
            try:
                sub_result = await sub_agent.run_once(inp.get("args") or "执行该 skill 任务。")
                self.total_input_tokens += sub_result["tokens"]["input"]
                self.total_output_tokens += sub_result["tokens"]["output"]
                self.total_cache_read_tokens += sub_result["tokens"].get("cache_read", 0)
                self.total_cache_creation_tokens += sub_result["tokens"].get("cache_creation", 0)
                print_sub_agent_end("skill-fork", inp.get("skill_name", ""))
                return sub_result["text"] or "(Skill 无输出)"
            except Exception as e:
                print_sub_agent_end("skill-fork", inp.get("skill_name", ""))
                return f"Skill fork 错误: {e}"

        return f'[Skill "{inp.get("skill_name", "")}" 已激活]\n\n{result["prompt"]}'

    # ─── Plan 模式辅助函数 ──────────────────────────────────────

    def _generate_plan_file_path(self) -> str:
        d = Path.home() / ".claude" / "plans"
        d.mkdir(parents=True, exist_ok=True)
        return str(d / f"plan-{self.session_id}.md")

    def _build_plan_mode_prompt(self) -> str:
        return f"""

# Plan 模式已启用

Plan 模式已启用。除了下面这个 plan 文件之外,绝对不要做任何修改、运行任何非只读工具,或对系统做任何变更。

## Plan 文件: {self._plan_file_path}
使用 write_file 或 edit_file 把你的计划增量写入这个文件。这是你唯一被允许编辑的文件。

## 工作流
1. **探索**: 读代码以理解任务。使用 read_file、list_files、grep_search。
2. **设计**: 设计你的实现方案。如果任务复杂,使用 agent 工具(type="plan")。
3. **写计划**: 将结构化的计划写入 plan 文件,包含:
   - **背景**: 为什么需要这个改动
   - **步骤**: 实现步骤,标出关键文件路径
   - **验证**: 如何测试这些改动
4. **退出**: 当计划已准备好供用户审阅时,调用 exit_plan_mode。

重要: 当你的计划完成后,你必须调用 exit_plan_mode。不要让用户来批准 —— exit_plan_mode 会处理批准流程。"""

    async def _execute_plan_mode_tool(self, name: str) -> str:
        if name == "enter_plan_mode":
            if self.permission_mode == "plan":
                return "已经在 plan 模式中。"
            self._pre_plan_mode = self.permission_mode
            self.permission_mode = "plan"
            self._plan_file_path = self._generate_plan_file_path()
            print_info("已进入 plan 模式(只读)。Plan 文件: " + self._plan_file_path)
            # 文案直接作为 tool_result 回写(message 通道),不碰 system 前缀。
            return (
                f"已进入 plan 模式。当前为只读模式。\n"
                f"{self._build_plan_mode_prompt()}"
            )

        if name == "exit_plan_mode":
            if self.permission_mode != "plan":
                return "当前不在 plan 模式。"
            plan_content = "(未找到 plan 文件)"
            if self._plan_file_path and Path(self._plan_file_path).exists():
                plan_content = Path(self._plan_file_path).read_text()

            # 交互式审批流程
            if self._plan_approval_fn:
                result = await self._plan_approval_fn(plan_content)
                choice = result.get("choice", "manual-execute")

                if choice == "keep-planning":
                    feedback = result.get("feedback") or "请修改计划。"
                    return (
                        f"用户拒绝了计划,希望继续规划。\n\n"
                        f"用户反馈: {feedback}\n\n"
                        f"请根据反馈修改你的计划。完成后再次调用 exit_plan_mode。"
                    )

                # 用户已批准 —— 决定目标模式
                if choice == "clear-and-execute":
                    target_mode = "acceptEdits"
                elif choice == "execute":
                    target_mode = "acceptEdits"
                else:  # manual-execute
                    target_mode = self._pre_plan_mode or "default"

                # 退出 plan 模式
                self.permission_mode = target_mode
                self._pre_plan_mode = None
                self._pending_plan_notice = None
                saved_plan_path = self._plan_file_path
                self._plan_file_path = None

                if choice == "clear-and-execute":
                    self.backend.clear_history()
                    self._context_cleared = True
                    print_info(f"计划已批准。上下文已清空,在 {target_mode} 模式下执行。")
                    return (
                        f"用户已批准计划。上下文已清空。权限模式: {target_mode}\n\n"
                        f"Plan 文件: {saved_plan_path}\n\n"
                        f"## 已批准的计划:\n{plan_content}\n\n"
                        f"开始实施。"
                    )

                print_info(f"计划已批准。在 {target_mode} 模式下执行。")
                return (
                    f"用户已批准计划。权限模式: {target_mode}\n\n"
                    f"## 已批准的计划:\n{plan_content}\n\n"
                    f"开始实施。"
                )

            # 兜底: 没有审批函数(例如子 Agent)
            self.permission_mode = self._pre_plan_mode or "default"
            self._pre_plan_mode = None
            self._plan_file_path = None
            self._pending_plan_notice = None
            print_info("已退出 plan 模式。已恢复为 " + self.permission_mode + " 模式。")
            return f"已退出 plan 模式。权限模式已恢复为: {self.permission_mode}\n\n## 你的计划:\n{plan_content}"

        return f"未知的 plan 模式工具: {name}"

    async def _execute_agent_tool(self, inp: dict) -> str:
        agent_type = inp.get("type", "general")
        description = inp.get("description", "sub-agent task")
        prompt = inp.get("prompt", "")

        print_sub_agent_start(agent_type, description)

        try:
            config = get_sub_agent_config(agent_type)
            sub_agent = Agent(
                model=self.model,
                custom_system_prompt=config["system_prompt"],
                custom_tools=config["tools"],
                is_sub_agent=True,
                permission_mode="plan" if self.permission_mode == "plan" else "bypassPermissions",
                **self.backend.child_config(),
            )
            result = await sub_agent.run_once(prompt)
            self.total_input_tokens += result["tokens"]["input"]
            self.total_output_tokens += result["tokens"]["output"]
            self.total_cache_read_tokens += result["tokens"].get("cache_read", 0)
            self.total_cache_creation_tokens += result["tokens"].get("cache_creation", 0)
            print_sub_agent_end(agent_type, description)
            return result["text"] or "(子 Agent 无输出)"
        except Exception as e:
            print_sub_agent_end(agent_type, description)
            return f"子 Agent 错误: {e}"

    # ─── 主循环 ───────────────────────────────────────

    async def _chat(self, user_message: str) -> None:
        # plan 约束经 message 通道注入(手动 /plan 与初始 --plan 没有 tool_result
        # 通道),prepend 到本回合 user 文本,不改 system 前缀。
        if self._pending_plan_notice:
            user_message = f"{self._pending_plan_notice}\n\n---\n\n{user_message}"
            self._pending_plan_notice = None
        self.backend.append_user_text(user_message)
        # 仅在回合边界做自动压缩 —— 此时最后一条消息已是普通 user 文本,
        # backend 的 compact_conversation 切片就不会把上一回合 tool 调用对斩断。
        await self._check_and_compact()

        # 启动异步记忆预取(非阻塞,每个用户回合触发一次)
        memory_prefetch: MemoryPrefetch | None = None
        if not self.is_sub_agent:
            sq = self.backend.side_query()
            if sq:
                memory_prefetch = start_memory_prefetch(
                    user_message, sq,
                    self._already_surfaced_memories, self._session_memory_bytes,
                )

        while True:
            if self._aborted:
                break

            self.backend.run_compression_pipeline()
            self._consume_memory_prefetch(memory_prefetch)

            self.sink.spinner_start()

            # ── 流式工具执行 ──────────────────────────────
            # 流式过程中每完成一个 tool_use 块,若并发安全且权限放行,立即启动
            # 执行 —— 这样工具会在模型还在生成时同步跑起来。OpenAI 后端没有
            # 对应事件,会自动忽略该回调。
            early_executions: dict[str, asyncio.Task] = {}

            def _on_tool_block(tu: NormalizedToolUse):
                if tu.name in CONCURRENCY_SAFE_TOOLS:
                    perm = check_permission(tu.name, tu.input, self.permission_mode, self._plan_file_path)
                    if perm["action"] == "allow":
                        task = asyncio.create_task(self._execute_tool_call(tu.name, tu.input))
                        early_executions[tu.id] = task

            def _on_attempt_retry():
                # 上一次 attempt 提前启动的 task 在新一轮里没人 await。
                _cancel_pending_early(early_executions)

            tool_uses, usage = await self.backend.stream(
                tools=get_active_tool_definitions(inject_agent_types(self.tools)),
                thinking_mode=self._thinking_mode,
                on_tool_block_complete=_on_tool_block,
                on_attempt_retry=_on_attempt_retry,
            )

            self.sink.spinner_stop()

            self.total_input_tokens += usage["input"]
            self.total_output_tokens += usage["output"]
            self.total_cache_read_tokens += usage.get("cache_read", 0)
            self.total_cache_creation_tokens += usage.get("cache_creation", 0)

            if not tool_uses:
                self.sink.turn_cost(
                    self.total_input_tokens,
                    self.total_output_tokens,
                    self._get_current_cost_usd(),
                    self.total_cache_read_tokens,
                )
                break

            self.current_turns += 1
            budget = self._check_budget()
            if budget["exceeded"]:
                print_info(f"超出预算: {budget['reason']}")
                break

            # 兜底批量并行: stream 期间没启动的并发安全工具,这里再扫一遍。
            # Anthropic 后端在 content_block_stop 已经早启动过了;OpenAI 后端
            # 没有等价事件,流式回调形同空操作,所有并发安全工具都靠这里启动,
            # 才能复现原 _chat_openai 的 asyncio.gather 行为。
            for tu in tool_uses:
                if tu.id in early_executions or tu.name not in CONCURRENCY_SAFE_TOOLS:
                    continue
                perm = check_permission(tu.name, tu.input, self.permission_mode, self._plan_file_path)
                if perm["action"] == "allow":
                    early_executions[tu.id] = asyncio.create_task(
                        self._execute_tool_call(tu.name, tu.input)
                    )

            # 处理工具调用: 已提前启动的直接 await;其余的走权限检查 + 执行
            tool_results: list[ToolResult] = []
            context_break = False
            for tu in tool_uses:
                if context_break or self._aborted:
                    break
                print_tool_call(tu.name, tu.input)

                early_task = early_executions.pop(tu.id, None)
                if early_task is not None:
                    raw = await early_task
                else:
                    perm = check_permission(tu.name, tu.input, self.permission_mode, self._plan_file_path)
                    if perm["action"] == "deny":
                        print_info(f"已拒绝: {perm.get('message', '')}")
                        tool_results.append(ToolResult(tu.id, f"动作被拒绝: {perm.get('message', '')}"))
                        continue
                    if perm["action"] == "confirm" and perm.get("message") and perm["message"] not in self._confirmed_paths:
                        confirmed = await confirm_dangerous(perm["message"])
                        if not confirmed:
                            tool_results.append(ToolResult(tu.id, "用户拒绝了该动作。"))
                            continue
                        self._confirmed_paths.add(perm["message"])
                    raw = await self._execute_tool_call(tu.name, tu.input)

                # 工具结果可能带孤立代理字符(非法 UTF-8 文件名经 os.listdir
                # 的 surrogateescape、MCP 结果等),入历史前清洗
                raw = sanitize_surrogates(raw)
                res = self._persist_large_result(tu.name, raw)
                print_tool_result(tu.name, res)

                if self._context_cleared:
                    self._context_cleared = False
                    self.backend.append_user_after_context_clear(res)
                    context_break = True
                    break
                tool_results.append(ToolResult(tu.id, res))

            if not context_break and tool_results:
                self.backend.append_tool_results(tool_results)
            # context_break / _aborted 时 for 循环提前退出,可能还有 early task
            # 没人 await。清理掉,避免孤儿。
            if early_executions:
                _cancel_pending_early(early_executions)
            self._context_cleared = False

    # ─── 记忆预取消费 ─────────────────────────────

    def _consume_memory_prefetch(self, prefetch: MemoryPrefetch | None) -> None:
        if not prefetch or not prefetch.settled or prefetch.consumed:
            return
        prefetch.consumed = True
        try:
            memories = prefetch.task.result()
            if memories:
                injection = format_memories_for_injection(memories)
                self.backend.append_user_text_inline_or_new(injection)
                for m in memories:
                    self._already_surfaced_memories.add(m.path)
                    self._session_memory_bytes += len(m.content.encode())
        except Exception:
            pass  # 预取错误已在 memory 模块里记录
