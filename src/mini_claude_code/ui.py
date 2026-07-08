"""终端 UI 渲染 —— 彩色输出、spinner、工具显示。"""

from __future__ import annotations

import sys
import threading
import time

from rich.console import Console

console = Console(highlight=False)

# ─── 基础输出 ──────────────────────────────────────────


def print_welcome() -> None:
    console.print("\n  [bold cyan]Mini Claude Code[/bold cyan][dim] —— 极简编码 Agent[/dim]\n")
    console.print("[dim]  输入你的请求,或输入 'exit' 退出。[/dim]")
    console.print("[dim]  命令: /clear /plan /cost /compact /memory /skills[/dim]\n")


def print_user_prompt() -> None:
    console.print("\n[bold green]> [/bold green]", end="")


def print_assistant_text(text: str) -> None:
    sys.stdout.write(text)
    sys.stdout.flush()


def print_tool_call(name: str, inp: dict) -> None:
    icon = _get_tool_icon(name)
    summary = _get_tool_summary(name, inp)
    console.print(f"\n  [yellow]{icon} {name}[/yellow][dim] {summary}[/dim]")


def print_tool_result(name: str, result: str) -> None:
    if (name in ("edit_file", "write_file")) and not result.startswith(("错误", "警告", "写入文件出错", "编辑文件出错")):
        _print_file_change_result(name, result)
        return
    max_len = 500
    truncated = result
    if len(result) > max_len:
        truncated = result[:max_len] + f"\n  ... (共 {len(result)} 字符)"
    lines = "\n".join("  " + l for l in truncated.split("\n"))
    console.print(f"[dim]{lines}[/dim]")


def _print_file_change_result(_name: str, result: str) -> None:
    lines = result.split("\n")
    console.print(f"[dim]  {lines[0]}[/dim]")

    max_display = 40
    content_lines = lines[1:]
    display_lines = content_lines[:max_display]

    for line in display_lines:
        if not line.strip():
            continue
        if line.startswith("@@"):
            console.print(f"[cyan]  {line}[/cyan]")
        elif line.startswith("- "):
            console.print(f"[red]  {line}[/red]")
        elif line.startswith("+ "):
            console.print(f"[green]  {line}[/green]")
        else:
            console.print(f"[dim]  {line}[/dim]")
    if len(content_lines) > max_display:
        console.print(f"[dim]  ... 还有 {len(content_lines) - max_display} 行[/dim]")


def print_error(msg: str) -> None:
    console.print(f"\n  [red]错误: {msg}[/red]")


def print_confirmation(command: str) -> None:
    console.print(f"\n  [yellow]⚠ 危险命令:[/yellow] [white]{command}[/white]")


async def confirm_dangerous(command: str) -> bool:
    print_confirmation(command)
    try:
        return input("  是否允许? (y/n): ").lower().startswith("y")
    except EOFError:
        return False


def print_divider() -> None:
    console.print(f"\n[dim]  {'─' * 50}[/dim]")


def print_cost(
    input_tokens: int,
    output_tokens: int,
    cost_usd: float | None = None,
    cache_read_tokens: int = 0,
) -> None:
    cache = f" / 缓存命中 {cache_read_tokens}" if cache_read_tokens else ""
    cost = f" (~${cost_usd:.4f})" if cost_usd is not None else ""
    console.print(
        f"\n[dim]  Token: 输入 {input_tokens} / 输出 {output_tokens}{cache}{cost}[/dim]"
    )


def print_retry(attempt: int, max_retries: int, reason: str) -> None:
    console.print(f"\n  [yellow]↻ 重试 {attempt}/{max_retries}: {reason}[/yellow]")


def print_info(msg: str) -> None:
    console.print(f"\n  [cyan]ℹ {msg}[/cyan]")


# ─── Spinner ──────────────────────────────────────────────

SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

_spinner_thread: threading.Thread | None = None
_spinner_stop = threading.Event()


def start_spinner(label: str = "思考中") -> None:
    global _spinner_thread
    if _spinner_thread is not None:
        return
    _spinner_stop.clear()

    def _run() -> None:
        frame = 0
        sys.stdout.write(f"\n  {SPINNER_FRAMES[0]} {label}...")
        sys.stdout.flush()
        while not _spinner_stop.is_set():
            time.sleep(0.08)
            frame = (frame + 1) % len(SPINNER_FRAMES)
            sys.stdout.write(f"\r  {SPINNER_FRAMES[frame]} {label}...")
            sys.stdout.flush()

    _spinner_thread = threading.Thread(target=_run, daemon=True)
    _spinner_thread.start()


def stop_spinner() -> None:
    global _spinner_thread
    if _spinner_thread is None:
        return
    _spinner_stop.set()
    _spinner_thread.join(timeout=1)
    _spinner_thread = None
    sys.stdout.write("\r\033[K")
    sys.stdout.flush()


# ─── Plan 审批显示 ──────────────────────────────────


def print_plan_for_approval(plan_content: str) -> None:
    console.print("\n  [cyan]━━━ 待审批的 Plan ━━━[/cyan]")
    lines = plan_content.split("\n")
    max_lines = 60
    for line in lines[:max_lines]:
        console.print(f"  [white]{line}[/white]")
    if len(lines) > max_lines:
        console.print(f"[dim]  ... 还有 {len(lines) - max_lines} 行[/dim]")
    console.print("  [cyan]━━━━━━━━━━━━━━━━━━━━━━━━[/cyan]\n")


def print_plan_approval_options() -> None:
    console.print("  [yellow]请选择:[/yellow]")
    console.print("    [white]1) 同意,清空上下文并执行[/white][dim] —— 全新开始,自动批准编辑[/dim]")
    console.print("    [white]2) 同意并执行[/white][dim] —— 保留上下文,自动批准编辑[/dim]")
    console.print("    [white]3) 同意,但手动批准编辑[/white][dim] —— 保留上下文,每次编辑都需要确认[/dim]")
    console.print("    [white]4) 不同意,继续规划[/white][dim] —— 提供反馈以修改计划[/dim]")


# ─── 子 Agent 显示 ──────────────────────────────────


def print_sub_agent_start(agent_type: str, description: str) -> None:
    console.print(f"\n  [magenta]┌─ 子 Agent [{agent_type}]: {description}[/magenta]")


def print_sub_agent_end(agent_type: str, _description: str) -> None:
    console.print(f"  [magenta]└─ 子 Agent [{agent_type}] 已完成[/magenta]")


# ─── 工具图标与摘要 ───────────────────────────────

_TOOL_ICONS = {
    "read_file": "📖",
    "write_file": "✏️",
    "edit_file": "🔧",
    "list_files": "📁",
    "grep_search": "🔍",
    "run_shell": "💻",
    "skill": "⚡",
    "agent": "🤖",
}


def _get_tool_icon(name: str) -> str:
    return _TOOL_ICONS.get(name, "🔨")


def _get_tool_summary(name: str, inp: dict) -> str:
    if name == "read_file":
        return inp.get("file_path", "")
    if name == "write_file":
        return inp.get("file_path", "")
    if name == "edit_file":
        return inp.get("file_path", "")
    if name == "list_files":
        return inp.get("pattern", "")
    if name == "grep_search":
        return f'"{inp.get("pattern", "")}" 于 {inp.get("path", ".")}'
    if name == "run_shell":
        cmd = inp.get("command", "")
        return cmd[:60] + "..." if len(cmd) > 60 else cmd
    if name == "skill":
        return inp.get("skill_name", "")
    if name == "agent":
        return f'[{inp.get("type", "general")}] {inp.get("description", "")}'
    return ""


# ─── 输出 sink ───────────────────────────────────────
# Agent 的输出通道抽象: 主 Agent 打到终端(ConsoleSink),子 Agent 捕获到
# 缓冲(BufferSink)。Agent 主循环无条件调 sink,不再散布 is_sub_agent 分支。


class OutputSink:
    """默认全部空操作 —— BufferSink 只需覆写 emit_text。"""

    def emit_text(self, text: str) -> None:
        pass

    def spinner_start(self) -> None:
        pass

    def spinner_stop(self) -> None:
        pass

    def turn_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float | None,
        cache_read_tokens: int,
    ) -> None:
        pass

    def session_end(self) -> None:
        pass


class ConsoleSink(OutputSink):
    def emit_text(self, text: str) -> None:
        print_assistant_text(text)

    def spinner_start(self) -> None:
        start_spinner()

    def spinner_stop(self) -> None:
        stop_spinner()

    def turn_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float | None,
        cache_read_tokens: int,
    ) -> None:
        print_cost(input_tokens, output_tokens, cost_usd, cache_read_tokens)

    def session_end(self) -> None:
        print_divider()


class BufferSink(OutputSink):
    def __init__(self) -> None:
        self._buffer: list[str] = []

    def emit_text(self, text: str) -> None:
        self._buffer.append(text)

    def take(self) -> str:
        """取走已捕获的全部文本并清空缓冲(run_once 结束时调用)。"""
        text = "".join(self._buffer)
        self._buffer = []
        return text
