"""Anthropic 后端实现。

消息形态: list[dict]，每条 user 消息的 content 可以是 str 或 content blocks
(其中 tool_result 是嵌在 user 列表里的 block,tool_use 是嵌在 assistant 列表
里的 block)。"""

from __future__ import annotations

import json
import time
from typing import Any, Awaitable, Callable

import anthropic

from ..tools import ToolDef
from .base import (
    Backend,
    CLEARED_PLACEHOLDER,
    KEEP_RECENT_RESULTS,
    NormalizedToolUse,
    SNIPPABLE_TOOLS,
    SNIP_PLACEHOLDER,
    SNIP_THRESHOLD,
    MICROCOMPACT_IDLE_S,
    ToolResult,
    estimate_tokens_from_messages,
    get_max_output_tokens,
    with_retry,
)


class AnthropicBackend(Backend):
    api_base = None  # Anthropic 用 SDK 默认 base url

    def __init__(
        self,
        *,
        model: str,
        system_prompt: str,
        emit_text: Callable[[str], None],
        stop_spinner: Callable[[], None],
        api_key: str | None = None,
        anthropic_base_url: str | None = None,
        on_retry: Callable[[int, int, str], None] | None = None,
    ):
        super().__init__(
            model=model,
            system_prompt=system_prompt,
            emit_text=emit_text,
            stop_spinner=stop_spinner,
            on_retry=on_retry,
        )
        self.messages: list[dict] = []
        # 留存原始配置供子 Agent 复用(SDK 内部存了但不可靠)
        self._api_key = api_key
        self._anthropic_base_url = anthropic_base_url
        kwargs: dict[str, Any] = {}
        if api_key:
            kwargs["api_key"] = api_key
        if anthropic_base_url:
            kwargs["base_url"] = anthropic_base_url
        self._client = anthropic.AsyncAnthropic(**kwargs)

    # ─── 系统提示词 ─────────────────────────────────

    def set_system_prompt(self, prompt: str) -> None:
        # Anthropic 在 messages.create 的 system 参数里读,所以只更新字段。
        self._system_prompt = prompt

    # ─── 消息追加 ──────────────────────────────────

    def append_user_text(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})

    def append_user_text_inline_or_new(self, text: str) -> None:
        last = self.messages[-1] if self.messages else None
        if last and last.get("role") == "user":
            content = last.get("content", "")
            if isinstance(content, str):
                last["content"] = content + "\n\n" + text
                return
            if isinstance(content, list):
                content.append({"type": "text", "text": text})
                return
        self.messages.append({"role": "user", "content": text})

    def append_tool_results(self, results: list[ToolResult]) -> None:
        blocks = [
            {"type": "tool_result", "tool_use_id": r.tool_use_id, "content": r.content}
            for r in results
        ]
        self.messages.append({"role": "user", "content": blocks})

    def append_user_after_context_clear(self, text: str) -> None:
        # 上下文已被清空,这里塞回一条普通 user 文本即可。
        self.messages.append({"role": "user", "content": text})

    def clear_history(self) -> None:
        self.messages = []
        self.last_input_token_count = 0

    # ─── 流式调用 ──────────────────────────────────

    async def stream(
        self,
        *,
        tools: list[ToolDef],
        thinking_mode: str,
        on_tool_block_complete: Callable[[NormalizedToolUse], None] | None,
        on_attempt_retry: Callable[[], None] | None = None,
    ) -> tuple[list[NormalizedToolUse], dict]:
        async def _do(attempt: int):
            if attempt > 0 and on_attempt_retry:
                on_attempt_retry()
            max_output = get_max_output_tokens(self.model)
            create_params: dict[str, Any] = {
                "model": self.model,
                "max_tokens": max_output if thinking_mode != "disabled" else 16384,
                "system": self._system_prompt,
                "tools": tools,
                "messages": self.messages,
            }
            if thinking_mode in ("adaptive", "enabled"):
                create_params["thinking"] = {"type": "enabled", "budget_tokens": max_output - 1}

            first_text = True
            tool_blocks_by_index: dict[int, dict] = {}

            async with self._client.messages.stream(**create_params) as stream:
                async for event in stream:
                    if not hasattr(event, "type"):
                        continue

                    if event.type == "content_block_start":
                        cb = getattr(event, "content_block", None)
                        if cb and getattr(cb, "type", None) == "tool_use":
                            tool_blocks_by_index[event.index] = {
                                "id": cb.id, "name": cb.name, "input_json": "",
                            }

                    elif event.type == "content_block_delta":
                        delta = event.delta
                        if hasattr(delta, "text"):
                            if first_text:
                                self._stop_spinner()
                                self._emit_text("\n")
                                first_text = False
                            self._emit_text(delta.text)
                        elif hasattr(delta, "thinking"):
                            if first_text:
                                self._stop_spinner()
                                self._emit_text("\n  [思考] ")
                                first_text = False
                            self._emit_text(delta.thinking)
                        elif hasattr(delta, "partial_json"):
                            tb = tool_blocks_by_index.get(event.index)
                            if tb:
                                tb["input_json"] += delta.partial_json

                    elif event.type == "content_block_stop":
                        tb = tool_blocks_by_index.pop(event.index, None)
                        if tb and on_tool_block_complete:
                            try:
                                parsed = json.loads(tb["input_json"] or "{}")
                            except Exception:
                                parsed = {}
                            on_tool_block_complete(NormalizedToolUse(
                                id=tb["id"], name=tb["name"], input=parsed,
                            ))

                final_message = await stream.get_final_message()

            # 过滤 thinking 块,不持久化
            final_message.content = [b for b in final_message.content if b.type != "thinking"]
            return final_message

        response = await with_retry(_do, on_retry=self._on_retry)

        # 写回 assistant 消息(消除外部 append_assistant 的需要)
        self.messages.append({
            "role": "assistant",
            "content": [self._block_to_dict(b) for b in response.content],
        })

        # 归一化 tool_uses
        tool_uses: list[NormalizedToolUse] = []
        for b in response.content:
            if b.type == "tool_use":
                inp = dict(b.input) if hasattr(b.input, "items") else b.input
                tool_uses.append(NormalizedToolUse(id=b.id, name=b.name, input=inp))

        usage = {
            "input": response.usage.input_tokens,
            "output": response.usage.output_tokens,
        }
        self.last_input_token_count = response.usage.input_tokens

        self.last_api_call_time = time.time()

        return tool_uses, usage

    @staticmethod
    def _block_to_dict(block) -> dict:
        if block.type == "text":
            return {"type": "text", "text": block.text}
        if block.type == "tool_use":
            return {
                "type": "tool_use",
                "id": block.id,
                "name": block.name,
                "input": dict(block.input) if hasattr(block.input, "items") else block.input,
            }
        return {"type": block.type}

    # ─── 压缩 第 1 层: 限额 ──────────────────────

    def budget_tool_results(self, window: int) -> None:
        utilization = self.last_input_token_count / window if window else 0
        if utilization < 0.5:
            return
        budget = 15000 if utilization > 0.7 else 30000
        for msg in self.messages:
            if msg.get("role") != "user" or not isinstance(msg.get("content"), list):
                continue
            for block in msg["content"]:
                if (
                    isinstance(block, dict)
                    and block.get("type") == "tool_result"
                    and isinstance(block.get("content"), str)
                    and len(block["content"]) > budget
                ):
                    keep = (budget - 80) // 2
                    block["content"] = (
                        block["content"][:keep]
                        + f"\n\n[... 限额裁剪: 中间截掉了 {len(block['content']) - keep * 2} 个字符 ...]\n\n"
                        + block["content"][-keep:]
                    )

    # ─── 压缩 第 2 层: 裁剪过期结果 ────────────

    def snip_stale_results(self, window: int) -> None:
        utilization = self.last_input_token_count / window if window else 0
        if utilization < SNIP_THRESHOLD:
            return

        results = []
        for mi, msg in enumerate(self.messages):
            if msg.get("role") != "user" or not isinstance(msg.get("content"), list):
                continue
            for bi, block in enumerate(msg["content"]):
                if (
                    isinstance(block, dict)
                    and block.get("type") == "tool_result"
                    and isinstance(block.get("content"), str)
                    and block["content"] != SNIP_PLACEHOLDER
                ):
                    tool_info = self._find_tool_use_by_id(block.get("tool_use_id"))
                    if tool_info and tool_info["name"] in SNIPPABLE_TOOLS:
                        results.append({
                            "mi": mi, "bi": bi,
                            "name": tool_info["name"],
                            "file_path": tool_info.get("input", {}).get("file_path"),
                        })

        if len(results) <= KEEP_RECENT_RESULTS:
            return

        to_snip = set()
        seen_files: dict[str, list[int]] = {}
        for i, r in enumerate(results):
            if r["name"] == "read_file" and r.get("file_path"):
                seen_files.setdefault(r["file_path"], []).append(i)

        for indices in seen_files.values():
            if len(indices) > 1:
                for j in indices[:-1]:
                    to_snip.add(j)

        snip_before = len(results) - KEEP_RECENT_RESULTS
        for i in range(snip_before):
            to_snip.add(i)

        for idx in to_snip:
            r = results[idx]
            self.messages[r["mi"]]["content"][r["bi"]]["content"] = SNIP_PLACEHOLDER

    # ─── 压缩 第 3 层: Microcompact ──────────────

    def microcompact(self) -> None:
        if not self.last_api_call_time or (time.time() - self.last_api_call_time) < MICROCOMPACT_IDLE_S:
            return
        all_results = []
        for mi, msg in enumerate(self.messages):
            if msg.get("role") != "user" or not isinstance(msg.get("content"), list):
                continue
            for bi, block in enumerate(msg["content"]):
                if (
                    isinstance(block, dict)
                    and block.get("type") == "tool_result"
                    and isinstance(block.get("content"), str)
                    and block["content"] not in (SNIP_PLACEHOLDER, CLEARED_PLACEHOLDER)
                ):
                    all_results.append((mi, bi))
        clear_count = len(all_results) - KEEP_RECENT_RESULTS
        for i in range(max(0, clear_count)):
            mi, bi = all_results[i]
            self.messages[mi]["content"][bi]["content"] = CLEARED_PLACEHOLDER

    # ─── 压缩 第 4 层: 摘要 ─────────────────────

    async def compact_conversation(self) -> bool:
        # 不变式: 调用者必须保证最后一条消息是普通的 user 文本消息
        # (不是 tool_result)。下面会切掉它;如果是 tool_result,
        # 前面 assistant 的 tool_use 会变成孤立块,API 会拒绝这次
        # 摘要调用。
        if len(self.messages) < 4:
            return False
        last_user_msg = self.messages[-1]
        summary_resp = await self._client.messages.create(
            model=self.model,
            max_tokens=2048,
            system="你是一个会话摘要器。简洁但要保留重要细节。",
            messages=[
                *self.messages[:-1],
                {"role": "user", "content": "用简洁的一段话总结到目前为止的对话,保留关键决策、文件路径以及继续工作所需的上下文。"},
            ],
        )
        summary_text = (
            summary_resp.content[0].text
            if summary_resp.content and summary_resp.content[0].type == "text"
            else "暂无摘要。"
        )
        self.messages = [
            {"role": "user", "content": f"[此前对话的摘要]\n{summary_text}"},
            {"role": "assistant", "content": "已了解。我掌握了之前对话的上下文。请问要如何继续协助?"},
        ]
        if last_user_msg.get("role") == "user":
            self.messages.append(last_user_msg)
        self.last_input_token_count = 0
        return True

    # ─── 内部: 反查 tool_use 元信息 ─────────────

    def _find_tool_use_by_id(self, tool_use_id: str) -> dict | None:
        for msg in self.messages:
            if msg.get("role") != "assistant" or not isinstance(msg.get("content"), list):
                continue
            for block in msg["content"]:
                if (
                    isinstance(block, dict)
                    and block.get("type") == "tool_use"
                    and block.get("id") == tool_use_id
                ):
                    return {"name": block["name"], "input": block.get("input", {})}
        return None

    # ─── 记忆 side query ────────────────────────

    def side_query(self) -> Callable[[str, str], Awaitable[str]] | None:
        client = self._client
        model = self.model

        async def _sq(system: str, user_message: str) -> str:
            resp = await client.messages.create(
                model=model, max_tokens=256, system=system,
                messages=[{"role": "user", "content": user_message}],
            )
            return "".join(b.text for b in resp.content if b.type == "text")

        return _sq

    # ─── 子 Agent 配置 ──────────────────────────

    def child_config(self) -> dict:
        return {
            "api_base": None,
            "anthropic_base_url": self._anthropic_base_url,
            "api_key": self._api_key,
        }

    # ─── 会话持久化 ─────────────────────────────

    def serialize(self) -> dict:
        return {"anthropicMessages": self.messages, "openaiMessages": None}

    def restore(self, session_data: dict) -> bool:
        if session_data.get("anthropicMessages"):
            self.messages = session_data["anthropicMessages"]
            self.last_input_token_count = estimate_tokens_from_messages(self.messages)
            return True
        return False
