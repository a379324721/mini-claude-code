"""Backend 抽象 + 两套具体实现。"""

from .base import Backend, NormalizedToolUse, ToolResult, get_context_window
from .anthropic import AnthropicBackend
from .openai import OpenAIBackend

__all__ = [
    "Backend",
    "NormalizedToolUse",
    "ToolResult",
    "AnthropicBackend",
    "OpenAIBackend",
    "get_context_window",
    "make_backend",
]


def make_backend(
    *,
    model: str,
    system_prompt: str,
    emit_text,
    stop_spinner,
    api_base: str | None,
    anthropic_base_url: str | None,
    api_key: str | None,
    on_retry=None,
) -> Backend:
    """根据 api_base 是否提供选择后端。"""
    if api_base:
        return OpenAIBackend(
            model=model,
            system_prompt=system_prompt,
            emit_text=emit_text,
            stop_spinner=stop_spinner,
            api_base=api_base,
            api_key=api_key,
            on_retry=on_retry,
        )
    return AnthropicBackend(
        model=model,
        system_prompt=system_prompt,
        emit_text=emit_text,
        stop_spinner=stop_spinner,
        api_key=api_key,
        anthropic_base_url=anthropic_base_url,
        on_retry=on_retry,
    )
