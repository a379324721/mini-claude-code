"""系统提示词构造 —— 内嵌模板、变量插值、上下文收集。"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path

from .memory import build_memory_prompt_section
from .skills import build_skill_descriptions
from .subagent import build_agent_descriptions
from .tools import get_deferred_tool_names

# ─── 系统提示词模板（内嵌）──────────────────────

SYSTEM_PROMPT_TEMPLATE = """\
你是 Mini Claude Code,一个轻量级编码助手 CLI。
你是一个交互式 Agent,帮助用户完成软件工程任务。请遵循以下指令,使用提供的工具协助用户。

# 系统
 - 你在工具调用之外输出的所有文本都会展示给用户。输出文本以与用户交流。
 - 工具在用户选定的权限模式下执行。当你尝试调用一个未被当前权限模式或权限设置自动放行的工具时,用户会被提示是否批准。

# 完成任务
 - 用户主要会请你完成软件工程任务,包括修 bug、加功能、重构代码、解释代码等。
 - 通常不要对你没读过的代码提出修改建议。如果用户问到或想让你改某个文件,先读它。
 - 如果某种方法失败了,先诊断原因再换路子——读错误信息、检查假设、做定向修复。
 - 小心不要引入安全漏洞,如命令注入、XSS、SQL 注入和其他 OWASP top 10 漏洞。如果发现自己写了不安全的代码,立即修复。优先写安全、可靠、正确的代码。
 - 避免过度工程化。只做直接被请求或明显必要的修改。保持方案简洁、聚焦。
   - 不要超出请求范围添加功能、重构代码或做"改进"。修一个 bug 不需要顺带清理周围代码。
   - 不要为不可能发生的场景添加错误处理、回退或校验。
 - 如果用户请求帮助,告诉他们可以输入 "exit" 退出,或使用 REPL 命令如 /clear、/cost、/compact、/memory、/skills。

# 谨慎执行动作
仔细考虑动作的可逆性和影响半径。一般来说,你可以自由执行本地、可逆的动作,比如编辑文件或运行测试。但对于难以撤销、会影响本地之外的共享系统、或者本身具有风险或破坏性的动作,继续之前先和用户确认。

# 使用你的工具
 - 当有合适的专用工具时,不要用 run_shell 执行命令。
   - 例如 读文件用 read_file,不要用 cat、head、tail 或 sed
   - run_shell 仅保留给必须 shell 执行的系统命令和终端操作。如果不确定且存在相关专用工具,默认使用专用工具,只在绝对必要时才回退到 run_shell。
 - 当任务匹配某个专用 Agent 的描述时,使用 `agent` 工具。Sub-agent 适合并行执行独立查询,或保护主上下文窗口不被过多结果污染,避免和 sub-agent 重复工作。

# 输出效率

重要: 直奔主题。先尝试最简单的方案,不要绕圈子。不要过度。务求精炼。

文本输出要简短、直接。先给答案或动作,再给推理。略去填充词、铺垫和多余的过渡。不要复述用户说过的话——直接去做。解释时只包含用户理解所必需的内容。

# 环境
工作目录: {{cwd}}
日期: {{date}}
平台: {{platform}}
Shell: {{shell}}
{{git_context}}
{{claude_md}}
{{memory}}
{{skills}}
{{agents}}
{{deferred_tools}}"""


import re as _re

# ─── @include 解析 ─────────────────────────────────────
# 解析 CLAUDE.md 中的 @./path、@~/path、@/path 引用。

_INCLUDE_RE = _re.compile(r"^@(\./[^\s]+|~/[^\s]+|/[^\s]+)$", _re.MULTILINE)
_MAX_INCLUDE_DEPTH = 5


def _resolve_includes(
    content: str,
    base_path: Path,
    visited: set[str] | None = None,
    depth: int = 0,
) -> str:
    if depth >= _MAX_INCLUDE_DEPTH:
        return content
    if visited is None:
        visited = set()

    def _replace(m: _re.Match) -> str:
        raw = m.group(1)
        if raw.startswith("~/"):
            resolved = Path.home() / raw[2:]
        elif raw.startswith("/"):
            resolved = Path(raw)
        else:
            resolved = base_path / raw
        resolved = resolved.resolve()
        key = str(resolved)
        if key in visited:
            return f"<!-- circular: {raw} -->"
        if not resolved.is_file():
            return f"<!-- not found: {raw} -->"
        try:
            visited.add(key)
            included = resolved.read_text()
            return _resolve_includes(included, resolved.parent, visited, depth + 1)
        except Exception:
            return f"<!-- error reading: {raw} -->"

    return _INCLUDE_RE.sub(_replace, content)


def _load_rules_dir(directory: Path) -> str:
    """加载 .claude/rules/ 目录下所有 .md 文件。"""
    rules_dir = directory / ".claude" / "rules"
    if not rules_dir.is_dir():
        return ""
    try:
        files = sorted(f for f in rules_dir.iterdir() if f.suffix == ".md" and f.is_file())
        if not files:
            return ""
        parts: list[str] = []
        for f in files:
            try:
                content = f.read_text()
                content = _resolve_includes(content, rules_dir)
                parts.append(f"<!-- rule: {f.name} -->\n{content}")
            except Exception:
                pass
        return "\n\n## Rules\n" + "\n\n".join(parts) if parts else ""
    except Exception:
        return ""


def load_claude_md() -> str:
    """从 cwd 向上层目录递归,收集所有 CLAUDE.md 并解析 @include。"""
    parts: list[str] = []
    d = Path.cwd().resolve()
    while True:
        f = d / "CLAUDE.md"
        if f.is_file():
            try:
                content = f.read_text()
                content = _resolve_includes(content, d)
                parts.insert(0, content)
            except Exception:
                pass
        parent = d.parent
        if parent == d:
            break
        d = parent
    # 加载 cwd 下的 .claude/rules/*.md
    rules = _load_rules_dir(Path.cwd())
    claude_md = ""
    if parts:
        claude_md = "\n\n# Project Instructions (CLAUDE.md)\n" + "\n\n---\n\n".join(parts)
    return claude_md + rules


def get_git_context() -> str:
    """获取 git 分支、近期提交和工作区状态。"""
    try:
        opts = {"encoding": "utf-8", "timeout": 3, "capture_output": True}
        branch = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], **opts).stdout.strip()
        log = subprocess.run(["git", "log", "--oneline", "-5"], **opts).stdout.strip()
        status = subprocess.run(["git", "status", "--short"], **opts).stdout.strip()
        result = f"\nGit branch: {branch}"
        if log:
            result += f"\nRecent commits:\n{log}"
        if status:
            result += f"\nGit status:\n{status}"
        return result
    except Exception:
        return ""


def build_system_prompt() -> str:
    """用内嵌模板和动态上下文拼出完整的系统提示词。"""
    from datetime import date
    today = date.today().isoformat()
    plat = f"{platform.system()} {platform.machine()}"
    shell = (os.environ.get("ComSpec") or "cmd.exe") if sys.platform == "win32" else os.environ.get("SHELL", "/bin/sh")
    git_context = get_git_context()
    claude_md = load_claude_md()
    memory_section = build_memory_prompt_section()
    skills_section = build_skill_descriptions()
    agent_section = build_agent_descriptions()

    deferred_names = get_deferred_tool_names()
    deferred_section = (
        f"\n\n以下延迟工具可通过 tool_search 调用: {', '.join(deferred_names)}。需要时使用 tool_search 获取它们的完整 schema。"
        if deferred_names else ""
    )

    replacements = {
        "{{cwd}}": str(Path.cwd()),
        "{{date}}": today,
        "{{platform}}": plat,
        "{{shell}}": shell,
        "{{git_context}}": git_context,
        "{{claude_md}}": claude_md,
        "{{memory}}": memory_section,
        "{{skills}}": skills_section,
        "{{agents}}": agent_section,
        "{{deferred_tools}}": deferred_section,
    }
    result = SYSTEM_PROMPT_TEMPLATE
    for key, value in replacements.items():
        result = result.replace(key, value)
    return result
