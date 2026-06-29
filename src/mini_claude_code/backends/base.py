"""Backend ABC —— 把双后端(Anthropic + OpenAI 兼容)差异封装在统一接口里。

Agent 主循环只面向这个抽象,消息格式、流式协议、压缩策略全部由具体 Backend
内部消化。"""

from __future__ import annotations

import asyncio
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from ..tools import ToolDef


def estimate_tokens_from_messages(messages: list[dict]) -> int:
    """粗略估算消息列表的 token 数(整体序列化字符数除以 4)。
    用于 restore 后让压缩管道有合理的初始水位估计 —— 比硬置为 0
    安全得多;首次 stream 返回真实 usage 后会被覆盖。"""
    try:
        return len(json.dumps(messages, default=str)) // 4
    except Exception:
        return 0


# ─── 模型上下文窗口 ──────────────────────────────────

MODEL_CONTEXT = {
    "claude-opus-4-6": 200000,
    "claude-sonnet-4-6": 200000,
    "claude-sonnet-4-20250514": 200000,
    "claude-haiku-4-5-20251001": 200000,
    "claude-opus-4-20250514": 200000,
    "gpt-4o": 128000,
    "gpt-4o-mini": 128000,
}


def get_context_window(model: str) -> int:
    return MODEL_CONTEXT.get(model, 200000)


# ─── Thinking 支持检测 ─────────────────────────────


def model_supports_thinking(model: str) -> bool:
    m = model.lower()
    if "claude-3-" in m or "3-5-" in m or "3-7-" in m:
        return False
    if "claude" in m and any(x in m for x in ("opus", "sonnet", "haiku")):
        return True
    return False


def model_supports_adaptive_thinking(model: str) -> bool:
    m = model.lower()
    return "opus-4-6" in m or "sonnet-4-6" in m


def get_max_output_tokens(model: str) -> int:
    m = model.lower()
    if "opus-4-6" in m:
        return 64000
    if "sonnet-4-6" in m:
        return 32000
    if any(x in m for x in ("opus-4", "sonnet-4", "haiku-4")):
        return 32000
    return 16384


# ─── 指数退避重试 ──────────────────────────────────


def is_retryable(error: Exception) -> bool:
    status = getattr(error, "status_code", None) or getattr(error, "status", None)
    if status in (429, 503, 529):
        return True
    msg = str(error)
    if "overloaded" in msg or "ECONNRESET" in msg or "ETIMEDOUT" in msg:
        return True
    return False


async def with_retry(fn, max_retries: int = 3, on_retry: Callable[[int, int, str], None] | None = None):
    for attempt in range(max_retries + 1):
        try:
            return await fn()
        except Exception as error:
            if attempt >= max_retries or not is_retryable(error):
                raise
            delay = min(1000 * (2 ** attempt), 30000) / 1000 + (hash(str(time.time())) % 1000) / 1000
            status = getattr(error, "status_code", None) or getattr(error, "status", None)
            reason = f"HTTP {status}" if status else (getattr(error, "code", None) or "网络错误")
            if on_retry:
                on_retry(attempt + 1, max_retries, reason)
            await asyncio.sleep(delay)


# ─── 多层压缩常量 ────────────────────────

SNIPPABLE_TOOLS = {"read_file", "grep_search", "list_files", "run_shell"}
SNIP_PLACEHOLDER = "[内容已裁剪 - 如需可重新读取]"
CLEARED_PLACEHOLDER = "[旧结果已清理]"
SNIP_THRESHOLD = 0.60
MICROCOMPACT_IDLE_S = 5 * 60  # 5 分钟
KEEP_RECENT_RESULTS = 3


# ─── 归一化类型 ──────────────────────────────────────


@dataclass
class NormalizedToolUse:
    """流式调用结束后,归一化后的工具调用项。"""
    id: str
    name: str
    input: dict


@dataclass
class ToolResult:
    """主循环回写给 Backend 的工具结果(批量)。"""
    tool_use_id: str
    content: str


# ─── Backend ABC ─────────────────────────────────────


class Backend(ABC):
    """两套 LLM 客户端共享的接口。

    生命周期:
      1. 构造时给定 model + 系统提示词 + emit_text/stop_spinner 回调
      2. 主循环通过 append_user_text / append_tool_results 维护历史
      3. stream() 每次返回归一化的 (tool_uses, usage),并内部已经把 assistant 消息写回 messages
      4. budget/snip/microcompact/compact 处理上下文窗口压力
      5. serialize/restore 让会话能存回磁盘
    """

    # 子类需要在 __init__ 里赋值 ↓
    messages: list[dict]
    api_base: str | None
    last_input_token_count: int
    last_api_call_time: float

    def __init__(
        self,
        *,
        model: str,
        system_prompt: str,
        emit_text: Callable[[str], None],
        stop_spinner: Callable[[], None],
        on_retry: Callable[[int, int, str], None] | None = None,
    ):
        self.model = model
        self._system_prompt = system_prompt
        self._emit_text = emit_text
        self._stop_spinner = stop_spinner
        self._on_retry = on_retry
        self.effective_window = get_context_window(model) - 20000
        self.last_input_token_count = 0
        self.last_api_call_time = 0.0

    # ── 系统提示词 ──
    @abstractmethod
    def set_system_prompt(self, prompt: str) -> None: ...

    # ── 消息追加 ──
    @abstractmethod
    def append_user_text(self, text: str) -> None: ...

    @abstractmethod
    def append_user_text_inline_or_new(self, text: str) -> None:
        """记忆注入:若最后一条是 user,把文本追加进去(兼容 str / list 两种 content);
        否则新建一条 user 文本消息。"""

    @abstractmethod
    def append_tool_results(self, results: list[ToolResult]) -> None: ...

    @abstractmethod
    def append_user_after_context_clear(self, text: str) -> None:
        """plan 模式批准 clear-and-execute 后,把工具结果作为新的 user 消息塞回来。"""

    def message_count(self) -> int:
        return len(self.messages)

    @abstractmethod
    def clear_history(self) -> None:
        """清空历史(保留系统提示词的归属交由具体后端处理)。"""

    # ── 流式调用 ──
    @abstractmethod
    async def stream(
        self,
        *,
        tools: list[ToolDef],
        thinking_mode: str,
        on_tool_block_complete: Callable[[NormalizedToolUse], None] | None,
    ) -> tuple[list[NormalizedToolUse], dict]:
        """发起一次流式 API 调用。

        返回 (tool_uses, usage),usage = {'input': int, 'output': int}。
        assistant 消息已由内部 append 到 messages。

        OpenAI 后端会忽略 on_tool_block_complete(无对应事件)。
        thinking_mode 由 Anthropic 后端读取,OpenAI 忽略。"""

    # ── 4 层压缩 ──
    @abstractmethod
    def budget_tool_results(self, window: int) -> None: ...

    @abstractmethod
    def snip_stale_results(self, window: int) -> None: ...

    @abstractmethod
    def microcompact(self) -> None: ...

    @abstractmethod
    async def compact_conversation(self) -> bool:
        """返回 True 表示真的执行了摘要,False 表示因消息太少早退、什么都没做。"""

    def run_compression_pipeline(self) -> None:
        """流水线: 限额 → 裁剪 → microcompact。回合边界调用。"""
        self.budget_tool_results(self.effective_window)
        self.snip_stale_results(self.effective_window)
        self.microcompact()

    # ── 记忆 side query ──
    @abstractmethod
    def side_query(self) -> Callable[[str, str], Awaitable[str]] | None:
        """返回一个用于记忆召回的轻量调用,签名 async (system, user) -> str。"""

    # ── 会话持久化 ──
    @abstractmethod
    def serialize(self) -> dict:
        """返回 {'anthropicMessages': [...]} 或 {'openaiMessages': [...]}。"""

    @abstractmethod
    def restore(self, session_data: dict) -> bool:
        """从 session JSON 还原 messages。返回 True 表示找到了本后端的消息
        并成功还原;返回 False 表示 session_data 里没有本后端的数据
        (跨后端切换、或老格式)。"""
