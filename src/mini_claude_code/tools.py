"""工具定义与执行 —— 10 个工具 + 5 种权限模式。
对齐 Claude Code 的工具系统: read_file、write_file、edit_file、list_files、
grep_search、run_shell、skill、enter/exit_plan_mode、agent。"""

from __future__ import annotations

import fnmatch
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from .memory import get_memory_dir
from .frontmatter import parse_frontmatter

# ─── 权限模式 ──────────────────────────────────────

PermissionMode = str  # "default" | "plan" | "acceptEdits" | "bypassPermissions" | "dontAsk"

READ_TOOLS = {"read_file", "list_files", "grep_search", "web_fetch"}
EDIT_TOOLS = {"write_file", "edit_file"}

# 并发安全的工具(只读、无副作用)可以并行执行
CONCURRENCY_SAFE_TOOLS = {"read_file", "list_files", "grep_search", "web_fetch"}

IS_WIN = sys.platform == "win32"

# ─── 类型别名 ──────────────────────────────────────────────

ToolDef = dict  # Anthropic tool schema dict

# ─── 工具定义 ───────────────────────────────────────

tool_definitions: list[ToolDef] = [
    {
        "name": "read_file",
        "description": "读取一个文件的内容。返回带行号的文件内容。",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "要读取的文件路径"},
            },
            "required": ["file_path"],
        },
    },
    {
        "name": "write_file",
        "description": "将内容写入文件。文件不存在则创建,存在则覆盖。",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "要写入的文件路径"},
                "content": {"type": "string", "description": "要写入文件的内容"},
            },
            "required": ["file_path", "content"],
        },
    },
    {
        "name": "edit_file",
        "description": "通过精确字符串匹配来编辑文件。old_string 必须完全匹配(包括空白和缩进)。",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "要编辑的文件路径"},
                "old_string": {"type": "string", "description": "要查找并替换的精确字符串"},
                "new_string": {"type": "string", "description": "替换成的新字符串"},
            },
            "required": ["file_path", "old_string", "new_string"],
        },
    },
    {
        "name": "list_files",
        "description": "按 glob 模式列出文件。返回匹配的文件路径。",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": '匹配文件的 glob 模式(例如 "**/*.ts"、"src/**/*")'},
                "path": {"type": "string", "description": "搜索起始目录。默认为当前目录。"},
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "grep_search",
        "description": "在文件中搜索某个模式。返回带文件路径与行号的匹配行。",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "要搜索的正则模式"},
                "path": {"type": "string", "description": "搜索的目录或文件。默认为当前目录。"},
                "include": {"type": "string", "description": '要纳入搜索的文件 glob(例如 "*.ts"、"*.py")'},
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "run_shell",
        "description": "执行一条 shell 命令并返回输出。用于跑测试、装包、git 操作等。",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "要执行的 shell 命令"},
                "timeout": {"type": "number", "description": "超时时间(毫秒,默认 30000)"},
            },
            "required": ["command"],
        },
    },
    {
        "name": "skill",
        "description": "按名称调用一个已注册的 skill。skill 是从 .claude/skills/ 加载的提示词模板。返回 skill 解析后的提示词供你遵循。",
        "input_schema": {
            "type": "object",
            "properties": {
                "skill_name": {"type": "string", "description": "要调用的 skill 名称"},
                "args": {"type": "string", "description": "传给 skill 的可选参数"},
            },
            "required": ["skill_name"],
        },
    },
    {
        "name": "web_fetch",
        "description": "抓取一个 URL 并以文本形式返回内容。对 HTML 页面会剥离标签返回可读文本。对 JSON/文本响应,内容直接返回。",
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "要抓取的 URL"},
                "max_length": {"type": "number", "description": "内容最大字符长度(默认 50000)"},
            },
            "required": ["url"],
        },
    },
    {
        "name": "enter_plan_mode",
        "description": "进入 plan 模式,切换到只读的规划阶段。在 plan 模式下,你只能读文件,以及写入 plan 文件。",
        "input_schema": {"type": "object", "properties": {}},
        "deferred": True,
    },
    {
        "name": "exit_plan_mode",
        "description": "在 plan 文件写完后退出 plan 模式。",
        "input_schema": {"type": "object", "properties": {}},
        "deferred": True,
    },
    {
        "name": "agent",
        "description": "启动一个子 Agent 自动完成任务。子 Agent 有独立上下文,只返回结果。类型: 'explore'(只读),'plan'(只读、结构化规划),'general'(全部工具)。",
        "input_schema": {
            "type": "object",
            "properties": {
                "description": {"type": "string", "description": "子 Agent 任务的简短描述(3-5 个词)"},
                "prompt": {"type": "string", "description": "给子 Agent 的详细任务指令"},
                "type": {"type": "string", "enum": ["explore", "plan", "general"], "description": "Agent 类型。默认: general"},
            },
            "required": ["description", "prompt"],
        },
    },
    # ─── 工具搜索(延迟工具加载器)─────────────────────
    {
        "name": "tool_search",
        "description": "按名称或关键字搜索可用工具。返回匹配的延迟工具的完整 schema 定义,以便你调用。",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "工具名或搜索关键字"},
            },
            "required": ["query"],
        },
    },
]

# ─── 延迟工具激活 ───────────────────────────────

_activated_tools: set[str] = set()


def reset_activated_tools() -> None:
    _activated_tools.clear()


def get_active_tool_definitions(all_tools: list[ToolDef] | None = None) -> list[ToolDef]:
    """返回当前生效的工具定义,排除尚未激活的延迟工具。
    会剥掉 'deferred' 字段,避免发到 API。"""
    tools = all_tools if all_tools is not None else tool_definitions
    return [
        {k: v for k, v in t.items() if k != "deferred"}
        for t in tools
        if not t.get("deferred") or t["name"] in _activated_tools
    ]


def get_deferred_tool_names(all_tools: list[ToolDef] | None = None) -> list[str]:
    """返回尚未激活的延迟工具的名称列表。"""
    tools = all_tools if all_tools is not None else tool_definitions
    return [t["name"] for t in tools if t.get("deferred") and t["name"] not in _activated_tools]


# ─── 工具执行 ─────────────────────────────────────────


def _read_file(inp: dict) -> str:
    try:
        content = Path(inp["file_path"]).read_text()
        lines = content.split("\n")
        numbered = "\n".join(f"{i+1:4d} | {line}" for i, line in enumerate(lines))
        return numbered
    except Exception as e:
        return f"读取文件出错: {e}"


def _write_file(inp: dict) -> str:
    try:
        path = Path(inp["file_path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(inp["content"])
        _auto_update_memory_index(str(path))
        lines = inp["content"].split("\n")
        line_count = len(lines)
        preview = "\n".join(f"{i+1:4d} | {l}" for i, l in enumerate(lines[:30]))
        trunc = f"\n  ... (共 {line_count} 行)" if line_count > 30 else ""
        return f"已成功写入 {inp['file_path']}(共 {line_count} 行)\n\n{preview}{trunc}"
    except Exception as e:
        return f"写入文件出错: {e}"


def _auto_update_memory_index(file_path: str) -> None:
    try:
        mem_dir = str(get_memory_dir())
        if file_path.startswith(mem_dir) and file_path.endswith(".md") and not file_path.endswith("MEMORY.md"):
            mem_path = Path(mem_dir)
            lines = ["# 记忆索引", ""]
            for f in sorted(mem_path.glob("*.md")):
                if f.name == "MEMORY.md":
                    continue
                try:
                    raw = f.read_text()
                    name_match = re.search(r"^name:\s*(.+)$", raw, re.MULTILINE)
                    type_match = re.search(r"^type:\s*(.+)$", raw, re.MULTILINE)
                    desc_match = re.search(r"^description:\s*(.+)$", raw, re.MULTILINE)
                    if name_match and type_match:
                        n = name_match.group(1).strip()
                        t = type_match.group(1).strip()
                        d = desc_match.group(1).strip() if desc_match else ""
                        lines.append(f"- **[{n}]({f.name})** ({t}) — {d}")
                except Exception:
                    pass
            (mem_path / "MEMORY.md").write_text("\n".join(lines))
    except Exception:
        pass


# ─── 编辑辅助: 引号归一化 + diff ───────────────


def _normalize_quotes(s: str) -> str:
    s = re.sub("[\u2018\u2019\u2032]", "'", s)
    s = re.sub('[\u201c\u201d\u2033]', '"', s)
    return s


def _find_actual_string(file_content: str, search_string: str) -> str | None:
    if search_string in file_content:
        return search_string
    norm_search = _normalize_quotes(search_string)
    norm_file = _normalize_quotes(file_content)
    idx = norm_file.find(norm_search)
    if idx != -1:
        return file_content[idx:idx + len(search_string)]
    return None


def _generate_diff(old_content: str, old_string: str, new_string: str) -> str:
    before_change = old_content.split(old_string)[0]
    line_num = before_change.count("\n") + 1
    old_lines = old_string.split("\n")
    new_lines = new_string.split("\n")

    parts = [f"@@ -{line_num},{len(old_lines)} +{line_num},{len(new_lines)} @@"]
    for l in old_lines:
        parts.append(f"- {l}")
    for l in new_lines:
        parts.append(f"+ {l}")
    return "\n".join(parts)


def _edit_file(inp: dict) -> str:
    try:
        path = Path(inp["file_path"])
        content = path.read_text()

        actual = _find_actual_string(content, inp["old_string"])
        if not actual:
            return f"错误: 在 {inp['file_path']} 中未找到 old_string"

        count = content.count(actual)
        if count > 1:
            return f"错误: old_string 在 {inp['file_path']} 中出现 {count} 次,必须唯一。"

        new_content = content.replace(actual, inp["new_string"], 1)
        path.write_text(new_content)

        diff = _generate_diff(content, actual, inp["new_string"])
        quote_note = "(经引号归一化后匹配)" if actual != inp["old_string"] else ""
        return f"已成功编辑 {inp['file_path']} {quote_note}\n\n{diff}"
    except Exception as e:
        return f"编辑文件出错: {e}"


def _list_files(inp: dict) -> str:
    try:
        base = Path(inp.get("path") or ".")
        pattern = inp["pattern"]
        files = []
        for p in base.glob(pattern):
            if p.is_file():
                rel = str(p.relative_to(base) if base != Path(".") else p)
                # 跳过 node_modules 和 .git
                if "node_modules" in rel or ".git" in rel.split(os.sep):
                    continue
                files.append(rel)
                if len(files) >= 200:
                    break
        if not files:
            return "未找到匹配该模式的文件。"
        result = "\n".join(files[:200])
        if len(files) > 200:
            result += f"\n... 还有 {len(files) - 200} 条未显示"
        return result
    except Exception as e:
        return f"列出文件出错: {e}"


def _grep_search(inp: dict) -> str:
    pattern = inp["pattern"]
    path = inp.get("path") or "."
    include = inp.get("include")

    # 先试系统 grep(Linux/macOS)
    if not IS_WIN:
        try:
            args = ["grep", "--line-number", "--color=never", "-r"]
            if include:
                args.append(f"--include={include}")
            args.extend(["--", pattern, path])
            result = subprocess.run(
                args, capture_output=True, text=True, timeout=10
            )
            if result.returncode == 1:
                return "未找到匹配。"
            if result.returncode == 0:
                lines = [l for l in result.stdout.split("\n") if l]
                output = "\n".join(lines[:100])
                if len(lines) > 100:
                    output += f"\n... 还有 {len(lines) - 100} 条匹配"
                return output
            # 非零退出(且不是 1)—— 跌落到 Python 兜底
        except Exception:
            pass  # 跌落到 Python 兜底

    # 纯 Python 兜底(Windows,或系统 grep 不可用)
    return _grep_python(pattern, path, include)


def _grep_python(pattern: str, directory: str, include: str | None) -> str:
    regex = re.compile(pattern)
    include_pattern = include
    matches: list[str] = []

    def walk(d: str) -> None:
        if len(matches) >= 200:
            return
        try:
            entries = os.listdir(d)
        except Exception:
            return
        for name in entries:
            if name.startswith(".") or name == "node_modules":
                continue
            full = os.path.join(d, name)
            if os.path.isdir(full):
                walk(full)
                continue
            if include_pattern and not fnmatch.fnmatch(name, include_pattern):
                continue
            try:
                text = Path(full).read_text(errors="replace")
                for i, line in enumerate(text.split("\n")):
                    if regex.search(line):
                        matches.append(f"{full}:{i+1}:{line}")
                        if len(matches) >= 200:
                            return
            except Exception:
                pass

    walk(directory)
    if not matches:
        return "未找到匹配。"
    output = "\n".join(matches[:100])
    if len(matches) > 100:
        output += f"\n... 还有 {len(matches) - 100} 条匹配"
    return output


def _run_shell(inp: dict) -> str:
    try:
        timeout_ms = inp.get("timeout", 30000)
        timeout_s = timeout_ms / 1000
        result = subprocess.run(
            inp["command"],
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        output = result.stdout or ""
        if result.returncode != 0:
            stderr = f"\nStderr: {result.stderr}" if result.stderr else ""
            stdout = f"\nStdout: {result.stdout}" if result.stdout else ""
            return f"命令失败(退出码 {result.returncode}){stdout}{stderr}"
        return output or "(无输出)"
    except subprocess.TimeoutExpired:
        return f"命令在 {inp.get('timeout', 30000)}ms 后超时"
    except Exception as e:
        return f"错误: {e}"


def _web_fetch(inp: dict) -> str:
    import urllib.request
    import urllib.error

    url = inp.get("url", "")
    max_length = inp.get("max_length", 50000)
    req = urllib.request.Request(url, headers={"User-Agent": "mini-claude/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            content_type = resp.headers.get("Content-Type", "")
            text = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        return f"HTTP 错误: {e.code} {e.reason}"
    except urllib.error.URLError as e:
        return f"抓取 {url} 出错: {e.reason}"
    except Exception as e:
        return f"抓取 {url} 出错: {e}"

    if "html" in content_type:
        text = re.sub(r"<script[\s\S]*?</script>", "", text, flags=re.IGNORECASE)
        text = re.sub(r"<style[\s\S]*?</style>", "", text, flags=re.IGNORECASE)
        text = re.sub(r"<[^>]*>", " ", text)
        text = text.replace("&nbsp;", " ").replace("&amp;", "&")
        text = text.replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", '"')
        text = re.sub(r"\s{2,}", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = text.strip()

    if len(text) > max_length:
        text = text[:max_length] + f"\n\n[... 已截断至 {max_length} 字符]"

    return text or "(空响应)"


# ─── 危险命令模式 ─────────────────────────────

DANGEROUS_PATTERNS = [
    re.compile(r"\brm\s"),
    re.compile(r"\bgit\s+(push|reset|clean|checkout\s+\.)"),
    re.compile(r"\bsudo\b"),
    re.compile(r"\bmkfs\b"),
    re.compile(r"\bdd\s"),
    re.compile(r">\s*/dev/"),
    re.compile(r"\bkill\b"),
    re.compile(r"\bpkill\b"),
    re.compile(r"\breboot\b"),
    re.compile(r"\bshutdown\b"),
    re.compile(r"\bdel\s", re.IGNORECASE),
    re.compile(r"\brmdir\s", re.IGNORECASE),
    re.compile(r"\bformat\s", re.IGNORECASE),
    re.compile(r"\btaskkill\s", re.IGNORECASE),
    re.compile(r"\bRemove-Item\s", re.IGNORECASE),
    re.compile(r"\bStop-Process\s", re.IGNORECASE),
]


def is_dangerous(command: str) -> bool:
    return any(p.search(command) for p in DANGEROUS_PATTERNS)


# ─── 权限规则(.claude/settings.json)───────────────


def _parse_rule(rule: str) -> dict:
    m = re.match(r"^([a-z_]+)\((.+)\)$", rule)
    if m:
        return {"tool": m.group(1), "pattern": m.group(2)}
    return {"tool": rule, "pattern": None}


def _load_settings(file_path: Path) -> dict | None:
    if not file_path.exists():
        return None
    try:
        return json.loads(file_path.read_text())
    except Exception:
        return None


_cached_rules: dict | None = None


def load_permission_rules() -> dict:
    global _cached_rules
    if _cached_rules is not None:
        return _cached_rules

    allow: list[dict] = []
    deny: list[dict] = []

    user_settings = _load_settings(Path.home() / ".claude" / "settings.json")
    project_settings = _load_settings(Path.cwd() / ".claude" / "settings.json")

    for settings in [user_settings, project_settings]:
        if not settings or "permissions" not in settings:
            continue
        perms = settings["permissions"]
        for r in perms.get("allow", []):
            allow.append(_parse_rule(r))
        for r in perms.get("deny", []):
            deny.append(_parse_rule(r))

    _cached_rules = {"allow": allow, "deny": deny}
    return _cached_rules


def _matches_rule(rule: dict, tool_name: str, inp: dict) -> bool:
    if rule["tool"] != tool_name:
        return False
    if rule["pattern"] is None:
        return True

    value = ""
    if tool_name == "run_shell":
        value = inp.get("command", "")
    elif "file_path" in inp:
        value = inp["file_path"]
    else:
        return True

    pattern = rule["pattern"]
    if pattern.endswith("*"):
        return value.startswith(pattern[:-1])
    return value == pattern


def _check_permission_rules(tool_name: str, inp: dict) -> str | None:
    rules = load_permission_rules()
    for rule in rules["deny"]:
        if _matches_rule(rule, tool_name, inp):
            return "deny"
    for rule in rules["allow"]:
        if _matches_rule(rule, tool_name, inp):
            return "allow"
    return None


def check_permission(
    tool_name: str,
    inp: dict,
    mode: str = "default",
    plan_file_path: str | None = None,
) -> dict:
    """返回 {"action": "allow"|"deny"|"confirm", "message": ...}"""
    if mode == "bypassPermissions":
        return {"action": "allow"}

    rule_result = _check_permission_rules(tool_name, inp)
    if rule_result == "deny":
        return {"action": "deny", "message": f"权限规则拒绝了 {tool_name}"}
    if rule_result == "allow":
        return {"action": "allow"}

    if tool_name in READ_TOOLS:
        return {"action": "allow"}

    if mode == "plan":
        if tool_name in EDIT_TOOLS:
            file_path = inp.get("file_path") or inp.get("path")
            if plan_file_path and file_path == plan_file_path:
                return {"action": "allow"}
            return {"action": "deny", "message": f"plan 模式下被阻止: {tool_name}"}
        if tool_name == "run_shell":
            return {"action": "deny", "message": "plan 模式下禁止 shell 命令"}

    if tool_name in ("enter_plan_mode", "exit_plan_mode"):
        return {"action": "allow"}

    if mode == "acceptEdits" and tool_name in EDIT_TOOLS:
        return {"action": "allow"}

    needs_confirm = False
    confirm_message = ""

    if tool_name == "run_shell" and is_dangerous(inp.get("command", "")):
        needs_confirm = True
        confirm_message = inp.get("command", "")
    elif tool_name == "write_file" and not Path(inp.get("file_path", "")).exists():
        needs_confirm = True
        confirm_message = f"写入新文件: {inp.get('file_path', '')}"
    elif tool_name == "edit_file" and not Path(inp.get("file_path", "")).exists():
        needs_confirm = True
        confirm_message = f"编辑不存在的文件: {inp.get('file_path', '')}"

    if needs_confirm:
        if mode == "dontAsk":
            return {"action": "deny", "message": f"自动拒绝(dontAsk 模式): {confirm_message}"}
        return {"action": "confirm", "message": confirm_message}

    return {"action": "allow"}


# ─── 工具结果截断 ─────────────────────────────

MAX_RESULT_CHARS = 50000


def _truncate_result(result: str) -> str:
    if len(result) <= MAX_RESULT_CHARS:
        return result
    keep_each = (MAX_RESULT_CHARS - 60) // 2
    return (
        result[:keep_each]
        + f"\n\n[... 中间截掉了 {len(result) - keep_each * 2} 个字符 ...]\n\n"
        + result[-keep_each:]
    )


# ─── 执行一次工具调用 ────────────────────────────────────
# "agent" 和 "skill" 工具在 agent.py 里处理,避免循环依赖。


async def execute_tool(
    name: str, inp: dict, read_file_state: dict[str, float] | None = None
) -> str:
    # ─── 编辑前必须先读 + mtime 新鲜度检查 ───────────
    if name == "read_file":
        result = _read_file(inp)
        if read_file_state is not None and not result.startswith("读取文件出错"):
            abs_path = str(Path(inp["file_path"]).resolve())
            try:
                read_file_state[abs_path] = os.path.getmtime(abs_path)
            except OSError:
                pass
        return _truncate_result(result)

    if name in ("write_file", "edit_file") and read_file_state is not None:
        abs_path = str(Path(inp["file_path"]).resolve())
        if os.path.exists(abs_path):
            if abs_path not in read_file_state:
                verb = "写入" if name == "write_file" else "编辑"
                return f"错误: 在{verb}之前必须先读取这个文件。请先用 read_file 查看其当前内容。"
            if os.path.getmtime(abs_path) != read_file_state[abs_path]:
                verb = "写入" if name == "write_file" else "编辑"
                return f"警告: 你上次读取之后 {inp['file_path']} 被外部修改过。请先 read_file 再{verb}。"

    # tool_search: 激活延迟工具并返回其 schema
    if name == "tool_search":
        query = (inp.get("query") or "").lower()
        deferred = [t for t in tool_definitions if t.get("deferred")]
        matches = [
            t for t in deferred
            if query in t["name"].lower() or query in (t.get("description") or "").lower()
        ]
        if not matches:
            return "未找到匹配的延迟工具。"
        for m in matches:
            _activated_tools.add(m["name"])
        return json.dumps(
            [{"name": t["name"], "description": t.get("description", ""), "input_schema": t["input_schema"]} for t in matches],
            indent=2,
        )

    handlers: dict = {
        "write_file": _write_file,
        "edit_file": _edit_file,
        "list_files": _list_files,
        "grep_search": _grep_search,
        "run_shell": _run_shell,
        "web_fetch": _web_fetch,
    }
    handler = handlers.get(name)
    if not handler:
        return f"未知工具: {name}"
    result = _truncate_result(handler(inp))

    # 写/编辑成功后更新 mtime
    if name in ("write_file", "edit_file") and read_file_state is not None and not result.startswith(("错误", "警告", "写入文件出错", "编辑文件出错")):
        abs_path = str(Path(inp["file_path"]).resolve())
        try:
            read_file_state[abs_path] = os.path.getmtime(abs_path)
        except OSError:
            pass

    return result


def reset_permission_cache() -> None:
    global _cached_rules
    _cached_rules = None
