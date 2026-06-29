"""OpenAI 兼容后端实现。

消息形态: list[dict],system 提示词作为 messages[0]; tool 调用通过
assistant.tool_calls + role="tool" + tool_call_id 闭环。"""

from __future__ import annotations

import json
import time
from typing import Any, Awaitable, Callable

import openai

from ..tools import ToolDef
from .base import (
    Backend,
    CLEARED_PLACEHOLDER,
    KEEP_RECENT_RESULTS,
    MICROCOMPACT_IDLE_S,
    NormalizedToolUse,
    SNIP_PLACEHOLDER,
    SNIP_THRESHOLD,
    ToolResult,
    estimate_tokens_from_messages,
    with_retry,
)


def _to_openai_tools(tools: list[ToolDef]) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["input_schema"],
            },
        }
        for t in tools
    ]


class OpenAIBackend(Backend):
    def __init__(
        self,
        *,
        model: str,
        system_prompt: str,
        emit_text: Callable[[str], None],
        stop_spinner: Callable[[], None],
        api_base: str,
        api_key: str | None = None,
        on_retry: Callable[[int, int, str], None] | None = None,
    ):
        super().__init__(
            model=model,
            system_prompt=system_prompt,
            emit_text=emit_text,
            stop_spinner=stop_spinner,
            on_retry=on_retry,
        )
        self.api_base = api_base
        self._api_key = api_key
        self._client = openai.AsyncOpenAI(base_url=api_base, api_key=api_key)
        self.messages: list[dict] = [{"role": "system", "content": system_prompt}]

    # ─── 系统提示词 ─────────────────────────────────

    def set_system_prompt(self, prompt: str) -> None:
        self._system_prompt = prompt
        if self.messages and self.messages[0].get("role") == "system":
            self.messages[0]["content"] = prompt
        else:
            self.messages.insert(0, {"role": "system", "content": prompt})

    # ─── 消息追加 ──────────────────────────────────

    def append_user_text(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})

    def append_user_text_inline_or_new(self, text: str) -> None:
        last = self.messages[-1] if self.messages else None
        if last and last.get("role") == "user":
            last["content"] = (last.get("content") or "") + "\n\n" + text
            return
        self.messages.append({"role": "user", "content": text})

    def append_tool_results(self, results: list[ToolResult]) -> None:
        for r in results:
            self.messages.append({
                "role": "tool",
                "tool_call_id": r.tool_use_id,
                "content": r.content,
            })

    def append_user_after_context_clear(self, text: str) -> None:
        # OpenAI 协议下,context_clear 走的是丢弃 tool_calls 之后再补 user 文本
        self.messages.append({"role": "user", "content": text})

    def clear_history(self) -> None:
        # 保留 system 提示词
        self.messages = [{"role": "system", "content": self._system_prompt}]
        self.last_input_token_count = 0

    # ─── 流式调用 ──────────────────────────────────

    async def stream(
        self,
        *,
        tools: list[ToolDef],
        thinking_mode: str,  # noqa: ARG002 — OpenAI 后端忽略
        on_tool_block_complete: Callable[[NormalizedToolUse], None] | None,  # noqa: ARG002
        on_attempt_retry: Callable[[], None] | None = None,
    ) -> tuple[list[NormalizedToolUse], dict]:
        async def _do(attempt: int):
            if attempt > 0 and on_attempt_retry:
                on_attempt_retry()
            stream = await self._client.chat.completions.create(
                model=self.model,
                tools=_to_openai_tools(tools),
                messages=self.messages,
                stream=True,
                stream_options={"include_usage": True},
            )

            content = ""
            first_text = True
            tool_calls: dict[int, dict] = {}
            finish_reason = ""
            usage = None

            async for chunk in stream:
                if chunk.usage:
                    usage = {
                        "prompt_tokens": chunk.usage.prompt_tokens,
                        "completion_tokens": chunk.usage.completion_tokens,
                    }

                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta

                if delta and delta.content:
                    if first_text:
                        self._stop_spinner()
                        self._emit_text("\n")
                        first_text = False
                    self._emit_text(delta.content)
                    content += delta.content

                if delta and delta.tool_calls:
                    for tc in delta.tool_calls:
                        existing = tool_calls.get(tc.index)
                        if existing:
                            if tc.function and tc.function.arguments:
                                existing["arguments"] += tc.function.arguments
                        else:
                            tool_calls[tc.index] = {
                                "id": tc.id or "",
                                "name": (tc.function.name if tc.function else "") or "",
                                "arguments": (tc.function.arguments if tc.function else "") or "",
                            }

                if chunk.choices[0].finish_reason:
                    finish_reason = chunk.choices[0].finish_reason

            assembled = None
            if tool_calls:
                assembled = [
                    {"id": tc["id"], "type": "function",
                     "function": {"name": tc["name"], "arguments": tc["arguments"]}}
                    for _, tc in sorted(tool_calls.items())
                ]

            return {
                "message": {
                    "role": "assistant",
                    "content": content or None,
                    "tool_calls": assembled,
                },
                "finish_reason": finish_reason or "stop",
                "usage": usage,  # 可能是 None — 某些 OpenAI 兼容服务不 emit usage
            }

        response = await with_retry(_do, on_retry=self._on_retry)

        message = response["message"]
        usage = response["usage"]

        # 写回 assistant 消息
        self.messages.append(message)

        # 归一化 tool_uses
        tool_uses: list[NormalizedToolUse] = []
        for tc in message.get("tool_calls") or []:
            if tc.get("type") != "function":
                continue
            try:
                inp = json.loads(tc["function"]["arguments"])
            except Exception:
                inp = {}
            tool_uses.append(NormalizedToolUse(
                id=tc["id"], name=tc["function"]["name"], input=inp,
            ))

        self.last_api_call_time = time.time()

        if usage is None:
            # 服务端没给 usage(vLLM / LM Studio / 部分代理) —— 保留上一次的水位
            # 估计,别覆盖为 0,否则压缩管道阈值永远摸不到。
            return tool_uses, {"input": 0, "output": 0}

        self.last_input_token_count = usage["prompt_tokens"]
        return tool_uses, {
            "input": usage["prompt_tokens"],
            "output": usage["completion_tokens"],
        }

    # ─── 压缩 第 1 层: 限额 ──────────────────────

    def budget_tool_results(self, window: int) -> None:
        utilization = self.last_input_token_count / window if window else 0
        if utilization < 0.5:
            return
        budget = 15000 if utilization > 0.7 else 30000
        for msg in self.messages:
            if (
                msg.get("role") == "tool"
                and isinstance(msg.get("content"), str)
                and len(msg["content"]) > budget
            ):
                keep = (budget - 80) // 2
                msg["content"] = (
                    msg["content"][:keep]
                    + f"\n\n[... 限额裁剪: 中间截掉了 {len(msg['content']) - keep * 2} 个字符 ...]\n\n"
                    + msg["content"][-keep:]
                )

    # ─── 压缩 第 2 层: 裁剪过期结果 ────────────

    def snip_stale_results(self, window: int) -> None:
        utilization = self.last_input_token_count / window if window else 0
        if utilization < SNIP_THRESHOLD:
            return
        tool_msgs = []
        for i, msg in enumerate(self.messages):
            if (
                msg.get("role") == "tool"
                and isinstance(msg.get("content"), str)
                and msg["content"] != SNIP_PLACEHOLDER
            ):
                tool_msgs.append(i)
        if len(tool_msgs) <= KEEP_RECENT_RESULTS:
            return
        snip_count = len(tool_msgs) - KEEP_RECENT_RESULTS
        for i in range(snip_count):
            self.messages[tool_msgs[i]]["content"] = SNIP_PLACEHOLDER

    # ─── 压缩 第 3 层: Microcompact ──────────────

    def microcompact(self) -> None:
        if not self.last_api_call_time or (time.time() - self.last_api_call_time) < MICROCOMPACT_IDLE_S:
            return
        tool_msgs = []
        for i, msg in enumerate(self.messages):
            if (
                msg.get("role") == "tool"
                and isinstance(msg.get("content"), str)
                and msg["content"] not in (SNIP_PLACEHOLDER, CLEARED_PLACEHOLDER)
            ):
                tool_msgs.append(i)
        clear_count = len(tool_msgs) - KEEP_RECENT_RESULTS
        for i in range(max(0, clear_count)):
            self.messages[tool_msgs[i]]["content"] = CLEARED_PLACEHOLDER

    # ─── 压缩 第 4 层: 摘要 ─────────────────────

    async def compact_conversation(self) -> bool:
        # 不变式: 调用者必须保证最后一条消息是普通的 user 文本消息
        # (不是 `tool` 角色的结果消息)。理由同 Anthropic 后端: 切掉 tool result
        # 会让前面 assistant 的 tool_calls 变成孤立块。
        if len(self.messages) < 5:
            return False
        system_msg = self.messages[0]
        last_user_msg = self.messages[-1]
        summary_resp = await self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "你是一个会话摘要器。简洁但要保留重要细节。"},
                *self.messages[1:-1],
                {"role": "user", "content": "用简洁的一段话总结到目前为止的对话,保留关键决策、文件路径以及继续工作所需的上下文。"},
            ],
        )
        summary_text = summary_resp.choices[0].message.content or "暂无摘要。"
        self.messages = [
            system_msg,
            {"role": "user", "content": f"[此前对话的摘要]\n{summary_text}"},
            {"role": "assistant", "content": "已了解。我掌握了之前对话的上下文。请问要如何继续协助?"},
        ]
        if last_user_msg.get("role") == "user":
            self.messages.append(last_user_msg)
        self.last_input_token_count = 0
        return True

    # ─── 记忆 side query ────────────────────────

    def side_query(self) -> Callable[[str, str], Awaitable[str]] | None:
        client = self._client
        model = self.model

        async def _sq(system: str, user_message: str) -> str:
            resp = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_message},
                ],
            )
            return resp.choices[0].message.content or "" if resp.choices else ""

        return _sq

    # ─── 子 Agent 配置 ──────────────────────────

    def child_config(self) -> dict:
        return {
            "api_base": self.api_base,
            "anthropic_base_url": None,
            "api_key": self._api_key,
        }

    # ─── 会话持久化 ─────────────────────────────

    def serialize(self) -> dict:
        return {"anthropicMessages": None, "openaiMessages": self.messages}

    def restore(self, session_data: dict) -> bool:
        if session_data.get("openaiMessages"):
            self.messages = session_data["openaiMessages"]
            self.last_input_token_count = estimate_tokens_from_messages(self.messages)
            return True
        return False
