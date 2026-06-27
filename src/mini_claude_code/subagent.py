"""子 Agent 系统 —— fork-return 模式,内置 + 自定义 Agent 类型。
对齐 Claude Code 的 AgentTool: explore(只读)、plan(结构化)、general(全部工具),
以及通过 .claude/agents/*.md 定义的用户自定义 Agent。"""

from __future__ import annotations

from pathlib import Path

from .frontmatter import parse_frontmatter
from .tools import tool_definitions, ToolDef

# ─── 只读工具(供 explore 和 plan agent 使用)──────────

READ_ONLY_TOOLS = {"read_file", "list_files", "grep_search"}

EXPLORE_PROMPT = """你是 Mini Claude Code 的文件搜索专家,擅长全面地浏览和探索代码库。

=== 关键: 只读模式 —— 禁止任何文件修改 ===
这是一个只读探索任务。严禁:
- 创建新文件(不允许 write_file、touch 或任何形式的文件创建)
- 修改已有文件(不允许 edit_file 操作)
- 删除文件(不允许 rm 或任何删除)
- 运行任何改变系统状态的命令

你的职责完全是搜索和分析已有代码。

你的强项:
- 使用 glob 模式快速定位文件
- 使用强大的正则搜索代码和文本
- 读取并分析文件内容

指南:
- 用 list_files 做宽泛的文件模式匹配
- 用 grep_search 配合正则搜文件内容
- 已知具体路径时,用 read_file 读
- 根据调用方指定的彻底程度调整搜索策略

注意: 你的定位是快速 Agent,要尽快返回输出。为此你必须:
- 高效使用手头的工具: 聪明地搜索文件和实现
- 尽可能并行地发起多个 grep 和读文件的工具调用

请高效地完成用户的搜索请求,并清晰报告你的发现。"""

PLAN_PROMPT = """你是 Plan agent —— 一个只读的子 Agent,专门负责设计实现计划。

重要约束:
- 你处于只读模式。只能使用 read_file、list_files 和 grep_search。
- 不要尝试修改任何文件。

你的工作:
- 分析代码库以理解当前架构
- 设计一份分步的实现计划
- 标识需要修改的关键文件
- 考虑架构上的取舍

返回一份结构化计划,包含:
1. 当前状态的总结
2. 分步的实现步骤
3. 涉及实现的关键文件
4. 潜在风险或注意事项"""

GENERAL_PROMPT = """你是 Mini Claude Code 的一个 Agent。请根据用户的消息,使用可用工具完成任务。彻底完成任务——别镀金,但也别半途而废。完成后,用一份精炼的报告说明做了什么以及关键发现 —— 调用方会把它转给用户,所以只需保留要点。

你的强项:
- 在大型代码库里搜索代码、配置和模式
- 分析多个文件以理解系统架构
- 调研需要浏览大量文件的复杂问题
- 执行多步研究型任务

指南:
- 文件搜索: 不知道东西在哪时先宽泛地搜。已知具体路径时直接 read_file。
- 分析: 先广后精。如果第一次搜索没结果,换多种策略再试。
- 务求彻底: 检查多个位置,考虑不同命名风格,关注相关文件。
- 除非绝对必要,绝不创建新文件。始终优先编辑已有文件。"""

# ─── 自定义 Agent 发现 ─────────────────────────────────

_cached_custom_agents: dict[str, dict] | None = None


def _discover_custom_agents() -> dict[str, dict]:
    global _cached_custom_agents
    if _cached_custom_agents is not None:
        return _cached_custom_agents

    agents: dict[str, dict] = {}
    # 用户级(优先级较低)
    _load_agents_from_dir(Path.home() / ".claude" / "agents", agents)
    # 项目级(优先级较高,会覆盖)
    _load_agents_from_dir(Path.cwd() / ".claude" / "agents", agents)

    _cached_custom_agents = agents
    return agents


def _load_agents_from_dir(directory: Path, agents: dict[str, dict]) -> None:
    if not directory.is_dir():
        return
    for entry in directory.iterdir():
        if not entry.suffix == ".md":
            continue
        try:
            raw = entry.read_text()
            result = parse_frontmatter(raw)
            meta = result.meta
            name = meta.get("name") or entry.stem
            allowed_tools = None
            if "allowed-tools" in meta:
                allowed_tools = [s.strip() for s in meta["allowed-tools"].split(",")]
            agents[name] = {
                "name": name,
                "description": meta.get("description", ""),
                "allowed_tools": allowed_tools,
                "system_prompt": result.body,
            }
        except Exception:
            pass


# ─── 主配置函数 ───────────────────────────────────


def get_sub_agent_config(agent_type: str) -> dict:
    """返回给定 Agent 类型对应的 {system_prompt, tools}。"""
    custom = _discover_custom_agents().get(agent_type)
    if custom:
        if custom["allowed_tools"]:
            tools = [t for t in tool_definitions if t["name"] in custom["allowed_tools"]]
        else:
            tools = [t for t in tool_definitions if t["name"] != "agent"]
        return {"system_prompt": custom["system_prompt"], "tools": tools}

    read_only = [t for t in tool_definitions if t["name"] in READ_ONLY_TOOLS]

    if agent_type == "explore":
        return {"system_prompt": EXPLORE_PROMPT, "tools": read_only}
    elif agent_type == "plan":
        return {"system_prompt": PLAN_PROMPT, "tools": read_only}
    else:  # general
        return {"system_prompt": GENERAL_PROMPT, "tools": [t for t in tool_definitions if t["name"] != "agent"]}


# ─── 可用的 Agent 类型(供系统提示词使用)──────────────


def get_available_agent_types() -> list[dict[str, str]]:
    types = [
        {"name": "explore", "description": "快速的只读代码库搜索与探索"},
        {"name": "plan", "description": "只读分析,产出结构化的实现计划"},
        {"name": "general", "description": "拥有完整工具,执行独立任务"},
    ]
    for name, defn in _discover_custom_agents().items():
        types.append({"name": name, "description": defn["description"]})
    return types


def build_agent_descriptions() -> str:
    types = get_available_agent_types()
    if len(types) <= 3:
        return ""  # 仅有内置类型,系统提示词里已含

    custom = types[3:]
    lines = ["\n# 自定义 Agent 类型", ""]
    for t in custom:
        lines.append(f"- **{t['name']}**: {t['description']}")
    return "\n".join(lines)


def reset_agent_cache() -> None:
    global _cached_custom_agents
    _cached_custom_agents = None
