"""prompt caching 改动的纯逻辑测试 —— 不触网。

只覆盖 3 件易回归的事:
1. 缓存计价公式(命中 0.1×、写入 1.25×)。
2. cache_control 断点注入,且不污染调用方传入的对象。
3. 不变式: 进出 plan 模式绝不改 system 前缀(否则缓存失效)。

"缓存是否真命中" 无法离线断言(要真实 API usage),靠 README 的手动 smoke。
"""
from unittest.mock import MagicMock, patch

import pytest

from mini_claude_code.agent import Agent
from mini_claude_code.backends.anthropic import AnthropicBackend


@pytest.fixture(autouse=True)
def _no_real_sdk():
    # 这些测试只验证纯逻辑,不发请求 —— 用 mock 替掉真实 SDK client,
    # 既避免触网,也绕开测试机上的 SOCKS 代理环境变量导致的构造报错。
    with patch(
        "mini_claude_code.backends.anthropic.anthropic.AsyncAnthropic",
        return_value=MagicMock(),
    ):
        yield


def _make_backend(model="claude-opus-4-8"):
    return AnthropicBackend(
        model=model,
        system_prompt="sys",
        emit_text=lambda _t: None,
        stop_spinner=lambda: None,
        api_key="test-dummy",  # 仅构造 SDK,不发请求
    )


# ── 1. 计价 ────────────────────────────────────────────

def test_cost_includes_cache_discounts():
    b = _make_backend()  # opus: input 15 / output 75 per M
    # 1M 命中(0.1×15=1.5) + 1M 写入(1.25×15=18.75) + 0 普通 input + 0 output
    cost = b.estimate_cost_usd(0, 0, cache_read_tokens=1_000_000,
                               cache_creation_tokens=1_000_000)
    assert cost == 1.5 + 18.75


def test_cost_backward_compatible_without_cache():
    b = _make_backend()
    # 老签名调用(只给 input/output)仍可用,cache 默认 0
    assert b.estimate_cost_usd(1_000_000, 0) == 15.0


def test_cost_unknown_model_returns_none():
    assert _make_backend(model="some-random-model").estimate_cost_usd(1, 1) is None


# ── 2. 断点注入 ────────────────────────────────────────

def test_with_cache_breakpoint_marks_last_only():
    tools = [{"name": "a"}, {"name": "b"}]
    out = AnthropicBackend._with_cache_breakpoint(tools)
    assert out[-1]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in out[0]


def test_with_cache_breakpoint_does_not_mutate_input():
    tools = [{"name": "a"}, {"name": "b"}]
    AnthropicBackend._with_cache_breakpoint(tools)
    assert "cache_control" not in tools[-1]  # 原对象不被污染


def test_messages_breakpoint_wraps_str_content():
    msgs = [{"role": "user", "content": "hello"}]
    out = AnthropicBackend._messages_with_cache_breakpoint(msgs)
    block = out[-1]["content"][0]
    assert block["type"] == "text" and block["text"] == "hello"
    assert block["cache_control"] == {"type": "ephemeral"}
    # 原 message 不被改写(serialize / 压缩流水线依赖它)
    assert msgs[-1]["content"] == "hello"


def test_messages_breakpoint_marks_last_block_of_list():
    msgs = [{"role": "assistant",
             "content": [{"type": "text", "text": "x"},
                         {"type": "tool_use", "id": "1", "name": "t", "input": {}}]}]
    out = AnthropicBackend._messages_with_cache_breakpoint(msgs)
    assert out[-1]["content"][-1]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in out[-1]["content"][0]
    assert "cache_control" not in msgs[-1]["content"][-1]  # 未污染


def test_messages_breakpoint_empty():
    assert AnthropicBackend._messages_with_cache_breakpoint([]) == []


# ── 3. plan 模式不变式: system 前缀恒定 ────────────────

def _make_agent():
    return Agent(model="claude-opus-4-8", api_key="test-dummy")


def test_enter_plan_does_not_touch_system():
    a = _make_agent()
    base = a._base_system_prompt
    a.toggle_plan_mode()  # 进入 plan
    assert a.permission_mode == "plan"
    assert a._system_prompt == base          # system 没变
    assert a._pending_plan_notice           # 约束改走 message 通道
    a.toggle_plan_mode()  # 退出
    assert a._system_prompt == base
    assert a._pending_plan_notice is None


def test_pending_notice_prepended_then_cleared():
    a = _make_agent()
    a.toggle_plan_mode()
    notice = a._pending_plan_notice
    assert notice is not None
    # 模拟 _chat 的注入逻辑(不触网)
    user_msg = "do something"
    if a._pending_plan_notice:
        merged = f"{a._pending_plan_notice}\n\n---\n\n{user_msg}"
        a._pending_plan_notice = None
    assert notice in merged and user_msg in merged
    assert a._pending_plan_notice is None
