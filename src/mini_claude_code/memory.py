"""记忆系统 —— 4 类的基于文件的记忆,带 MEMORY.md 索引。
对齐 Claude Code 的记忆架构: 通过 sideQuery 做语义召回。"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .frontmatter import parse_frontmatter, format_frontmatter

# 发送一段提示词并返回模型文本响应的可调用对象。
# 签名: async (system: str, user_message: str) -> str
from typing import Callable
SideQueryFn = Callable[[str, str], Any]  # 实际上是 Awaitable[str]

# ─── 类型 ──────────────────────────────────────────────────

VALID_TYPES = {"user", "feedback", "project", "reference"}
MAX_INDEX_LINES = 200
MAX_INDEX_BYTES = 25000


class MemoryEntry:
    __slots__ = ("name", "description", "type", "filename", "content")

    def __init__(self, name: str, description: str, type: str, filename: str, content: str):
        self.name = name
        self.description = description
        self.type = type
        self.filename = filename
        self.content = content


# ─── 路径 ──────────────────────────────────────────────────


def _project_hash() -> str:
    return hashlib.sha256(str(Path.cwd()).encode()).hexdigest()[:16]


def get_memory_dir() -> Path:
    d = Path.home() / ".mini-claude" / "projects" / _project_hash() / "memory"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _get_index_path() -> Path:
    return get_memory_dir() / "MEMORY.md"


# ─── Slugify ────────────────────────────────────────────────


def _slugify(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", text.lower())
    s = s.strip("_")
    return s[:40]


# ─── CRUD 操作 ───────────────────────────────────────────────────


def list_memories() -> list[MemoryEntry]:
    d = get_memory_dir()
    entries: list[MemoryEntry] = []
    for f in sorted(d.glob("*.md")):
        if f.name == "MEMORY.md":
            continue
        try:
            result = parse_frontmatter(f.read_text())
            meta = result.meta
            if not meta.get("name") or not meta.get("type"):
                continue
            t = meta["type"] if meta["type"] in VALID_TYPES else "project"
            entries.append(MemoryEntry(
                name=meta["name"],
                description=meta.get("description", ""),
                type=t,
                filename=f.name,
                content=result.body,
            ))
        except Exception:
            pass
    # 按 mtime 倒序排
    entries.sort(key=lambda e: (d / e.filename).stat().st_mtime, reverse=True)
    return entries


def save_memory(name: str, description: str, type: str, content: str) -> str:
    d = get_memory_dir()
    filename = f"{type}_{_slugify(name)}.md"
    text = format_frontmatter({"name": name, "description": description, "type": type}, content)
    (d / filename).write_text(text)
    _update_memory_index()
    return filename


def delete_memory(filename: str) -> bool:
    filepath = get_memory_dir() / filename
    if not filepath.exists():
        return False
    filepath.unlink()
    _update_memory_index()
    return True


# ─── 索引 ──────────────────────────────────────────────────


def _update_memory_index() -> None:
    memories = list_memories()
    lines = ["# 记忆索引", ""]
    for m in memories:
        lines.append(f"- **[{m.name}]({m.filename})** ({m.type}) — {m.description}")
    _get_index_path().write_text("\n".join(lines))


def load_memory_index() -> str:
    index_path = _get_index_path()
    if not index_path.exists():
        return ""
    content = index_path.read_text()
    lines = content.split("\n")
    if len(lines) > MAX_INDEX_LINES:
        content = "\n".join(lines[:MAX_INDEX_LINES]) + "\n\n[... 已截断,条目过多 ...]"
    if len(content.encode()) > MAX_INDEX_BYTES:
        content = content[:MAX_INDEX_BYTES] + "\n\n[... 已截断,索引过大 ...]"
    return content


# ─── 记忆头(轻量扫描)──────────────────────

class MemoryHeader:
    __slots__ = ("filename", "file_path", "mtime_ms", "description", "type")

    def __init__(self, filename: str, file_path: str, mtime_ms: float,
                 description: str | None, type: str | None):
        self.filename = filename
        self.file_path = file_path
        self.mtime_ms = mtime_ms
        self.description = description
        self.type = type


MAX_MEMORY_FILES = 200
MAX_MEMORY_BYTES_PER_FILE = 4096
MAX_SESSION_MEMORY_BYTES = 60 * 1024  # 单次会话累计 60KB


def scan_memory_headers() -> list[MemoryHeader]:
    """扫描记忆目录 —— 只读 frontmatter(前 30 行)以提速。"""
    d = get_memory_dir()
    headers: list[MemoryHeader] = []
    for f in d.glob("*.md"):
        if f.name == "MEMORY.md":
            continue
        try:
            stat = f.stat()
            raw = f.read_text()
            first30 = "\n".join(raw.split("\n")[:30])
            result = parse_frontmatter(first30)
            meta = result.meta
            t = meta.get("type")
            headers.append(MemoryHeader(
                filename=f.name,
                file_path=str(f),
                mtime_ms=stat.st_mtime * 1000,
                description=meta.get("description"),
                type=t if t in VALID_TYPES else None,
            ))
        except Exception:
            pass
    headers.sort(key=lambda h: h.mtime_ms, reverse=True)
    return headers[:MAX_MEMORY_FILES]


def format_memory_manifest(headers: list[MemoryHeader]) -> str:
    """为语义选择器格式化 manifest: 每条记忆一行。"""
    lines = []
    for h in headers:
        tag = f"[{h.type}] " if h.type else ""
        ts = datetime.fromtimestamp(h.mtime_ms / 1000, tz=timezone.utc).isoformat()
        if h.description:
            lines.append(f"- {tag}{h.filename} ({ts}): {h.description}")
        else:
            lines.append(f"- {tag}{h.filename} ({ts})")
    return "\n".join(lines)


# ─── 记忆年龄 / 新鲜度 ────────────────────────────────

def memory_age(mtime_ms: float) -> str:
    days = max(0, int((time.time() * 1000 - mtime_ms) / 86_400_000))
    if days == 0:
        return "今天"
    if days == 1:
        return "昨天"
    return f"{days} 天前"


def memory_freshness_warning(mtime_ms: float) -> str:
    days = max(0, int((time.time() * 1000 - mtime_ms) / 86_400_000))
    if days <= 1:
        return ""
    return (f"这条记忆已经 {days} 天没更新了。记忆是某个时刻的观察,不是实时状态 —— "
            f"关于代码行为的断言可能已经过时。在作为事实使用前,请对照当前代码验证。")


# ─── 语义召回(sideQuery)────────────────────────────

SELECT_MEMORIES_PROMPT = """你正在为一个 AI 编码助手挑选与用户查询相关、能帮助它处理任务的记忆。你会收到用户的查询,以及一份可用记忆文件的列表(包含文件名和描述)。

返回一个 JSON 对象,字段 "selected_memories" 是一个数组,列出明显有用的记忆文件名(最多 5 个)。只基于名称和描述,纳入你确信会有帮助的记忆。
- 不确定是否有用的记忆,不要纳入。
- 如果没有明确有用的记忆,返回空数组。"""


class RelevantMemory:
    __slots__ = ("path", "content", "mtime_ms", "header")

    def __init__(self, path: str, content: str, mtime_ms: float, header: str):
        self.path = path
        self.content = content
        self.mtime_ms = mtime_ms
        self.header = header


async def select_relevant_memories(
    query: str,
    side_query: SideQueryFn,
    already_surfaced: set[str],
) -> list[RelevantMemory]:
    """让模型按语义挑选相关记忆。"""
    headers = scan_memory_headers()
    if not headers:
        return []

    candidates = [h for h in headers if h.file_path not in already_surfaced]
    if not candidates:
        return []

    manifest = format_memory_manifest(candidates)

    try:
        text = await side_query(
            SELECT_MEMORIES_PROMPT,
            f"查询: {query}\n\n可用记忆:\n{manifest}",
        )

        # 从响应里提取 JSON
        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            return []

        parsed = json.loads(match.group(0))
        selected_filenames = set(parsed.get("selected_memories", []))

        selected = [h for h in candidates if h.filename in selected_filenames][:5]

        result: list[RelevantMemory] = []
        for h in selected:
            content = Path(h.file_path).read_text()
            if len(content.encode()) > MAX_MEMORY_BYTES_PER_FILE:
                content = content[:MAX_MEMORY_BYTES_PER_FILE] + "\n\n[... 已截断,记忆文件过大 ...]"
            freshness = memory_freshness_warning(h.mtime_ms)
            header_text = (
                f"{freshness}\n\n记忆: {h.file_path}:" if freshness
                else f"记忆(保存于 {memory_age(h.mtime_ms)}): {h.file_path}:"
            )
            result.append(RelevantMemory(
                path=h.file_path, content=content,
                mtime_ms=h.mtime_ms, header=header_text,
            ))
        return result
    except Exception as e:
        if "cancel" in str(e).lower():
            return []
        print(f"[memory] 语义召回失败: {e}")
        return []


# ─── 预取句柄 ────────────────────────────────────────

class MemoryPrefetch:
    def __init__(self, task: asyncio.Task):
        self.task = task
        self.consumed = False

    @property
    def settled(self) -> bool:
        return self.task.done()


def start_memory_prefetch(
    query: str,
    side_query: SideQueryFn,
    already_surfaced: set[str],
    session_memory_bytes: int,
) -> MemoryPrefetch | None:
    """启动异步记忆预取。返回句柄供轮询结果。"""
    # 门槛: 仅多词输入
    if not re.search(r"\s", query.strip()):
        return None

    # 门槛: 会话预算
    if session_memory_bytes >= MAX_SESSION_MEMORY_BYTES:
        return None

    # 门槛: 必须存在记忆
    d = get_memory_dir()
    has_memories = any(f.suffix == ".md" and f.name != "MEMORY.md" for f in d.iterdir())
    if not has_memories:
        return None

    task = asyncio.create_task(
        select_relevant_memories(query, side_query, already_surfaced)
    )
    return MemoryPrefetch(task)


def format_memories_for_injection(memories: list[RelevantMemory]) -> str:
    """把召回的记忆格式化成可作为 user 消息内容注入的字符串。"""
    parts = []
    for m in memories:
        parts.append(f"<system-reminder>\n{m.header}\n\n{m.content}\n</system-reminder>")
    return "\n\n".join(parts)


# ─── 系统提示词章节 ──────────────────────────────────


def build_memory_prompt_section() -> str:
    index = load_memory_index()
    memory_dir = str(get_memory_dir())

    return f"""# 记忆系统

你在 `{memory_dir}` 拥有一个持久化、基于文件的记忆系统。

## 记忆类型
- **user**: 用户的角色、偏好、知识水平
- **feedback**: 来自用户的纠正和指引(包含 Why + 如何应用)
- **project**: 进行中的工作、目标、截止日期、决策
- **reference**: 指向外部资源的指针(URL、工具、看板)

## 如何保存记忆
使用 write_file 工具创建一个带 YAML frontmatter 的记忆文件:

```markdown
---
name: 记忆名称
description: 一行描述
type: user|feedback|project|reference
---
记忆内容。
```

保存到: `{memory_dir}/`
文件名格式: `{{type}}_{{slugified_name}}.md`

写入记忆目录时,MEMORY.md 索引会自动更新 —— 不要手动修改它。

## 不要保存什么
- 代码模式或架构(直接读代码即可)
- Git 历史(用 git log)
- CLAUDE.md 里已有的内容
- 临时任务细节

## 何时召回
当用户要求你记住或回忆,或先前的上下文看起来与当前任务相关时。
{chr(10) + "## 当前记忆索引" + chr(10) + index if index else chr(10) + "(尚未保存任何记忆。)"}"""
