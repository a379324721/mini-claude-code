
from __future__ import annotations

import argparse
import asyncio
import os
import signal
import sys

from dotenv import load_dotenv

# 自动从 cwd 向上找 .env 并加载,让 ANTHROPIC_API_KEY / OPENAI_API_KEY 等环境变量可读
load_dotenv()

from .agent import Agent
from .ui import print_welcome, print_user_prompt, print_error, print_info, print_plan_for_approval, print_plan_approval_options
from .session import load_session, get_latest_session_id
from .memory import list_memories
from .skills import discover_skills, resolve_skill_prompt, get_skill_by_name, execute_skill


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="mini-claude",
        description="Mini Claude Code —— 极简编码 Agent",
        add_help=False,
    )
    parser.add_argument("prompt", nargs="*", help="一次性提示词")
    parser.add_argument("--yolo", "-y", action="store_true", help="跳过所有确认提示")
    parser.add_argument("--plan", action="store_true", help="Plan 模式: 只读")
    parser.add_argument("--accept-edits", action="store_true", help="自动批准文件编辑")
    parser.add_argument("--dont-ask", action="store_true", help="自动拒绝需要确认的操作(适用于 CI)")
    parser.add_argument("--thinking", action="store_true", help="启用扩展 thinking")
    parser.add_argument("--model", "-m", default=None, help="使用的模型")
    parser.add_argument("--api-base", default=None, help="OpenAI 兼容 API 的 base URL")
    parser.add_argument("--resume", action="store_true", help="恢复上一次会话")
    parser.add_argument("--max-cost", type=float, default=None, help="最大花费(美元)")
    parser.add_argument("--max-turns", type=int, default=None, help="最大 Agent 回合数")
    parser.add_argument("--help", "-h", action="store_true", help="显示帮助")
    return parser.parse_args()


def _resolve_permission_mode(args: argparse.Namespace) -> str:
    if args.yolo:
        return "bypassPermissions"
    if args.plan:
        return "plan"
    if args.accept_edits:
        return "acceptEdits"
    if args.dont_ask:
        return "dontAsk"
    return "default"


async def run_repl(agent: Agent) -> None:
    """交互式 REPL 循环。"""

    async def plan_approval_fn(plan_content: str) -> dict:
        print_plan_for_approval(plan_content)
        print_plan_approval_options()
        while True:
            try:
                choice = input("  请选择(1-4): ").strip()
            except EOFError:
                return {"choice": "manual-execute"}
            if choice == "1":
                return {"choice": "clear-and-execute"}
            elif choice == "2":
                return {"choice": "execute"}
            elif choice == "3":
                return {"choice": "manual-execute"}
            elif choice == "4":
                try:
                    feedback = input("  反馈(需要调整什么): ").strip()
                except EOFError:
                    feedback = ""
                return {"choice": "keep-planning", "feedback": feedback or None}
            else:
                print("  无效选择。请输入 1、2、3 或 4。")

    agent.set_plan_approval_fn(plan_approval_fn)

    sigint_count = 0

    def handle_sigint(sig, frame):
        nonlocal sigint_count
        if agent._aborted is False and agent._output_buffer is not None:
            # Agent 正在处理中
            agent.abort()
            print("\n  (已中断)")
            sigint_count = 0
            print_user_prompt()
        else:
            sigint_count += 1
            if sigint_count >= 2:
                print("\n再见!\n")
                sys.exit(0)
            print("\n  再按一次 Ctrl+C 退出。")
            print_user_prompt()

    signal.signal(signal.SIGINT, handle_sigint)
    print_welcome()

    while True:
        print_user_prompt()
        try:
            line = input()
        except (EOFError, KeyboardInterrupt):
            print("\n再见!\n")
            break

        inp = line.strip()
        sigint_count = 0

        if not inp:
            continue
        if inp in ("exit", "quit"):
            print("\n再见!\n")
            break

        # REPL 命令
        if inp == "/clear":
            agent.clear_history()
            continue
        if inp == "/plan":
            agent.toggle_plan_mode()
            continue
        if inp == "/cost":
            agent.show_cost()
            continue
        if inp == "/compact":
            try:
                await agent.compact()
            except Exception as e:
                print_error(str(e))
            continue
        if inp == "/memory":
            memories = list_memories()
            if not memories:
                print_info("尚未保存任何记忆。")
            else:
                print_info(f"共 {len(memories)} 条记忆:")
                for m in memories:
                    print(f"    [{m.type}] {m.name} — {m.description}")
            continue
        if inp == "/skills":
            skills = discover_skills()
            if not skills:
                print_info("未发现 skills。请把 skill 加到 .claude/skills/<name>/SKILL.md")
            else:
                print_info(f"共 {len(skills)} 个 skill:")
                for s in skills:
                    tag = f"/{s.name}" if s.user_invocable else s.name
                    print(f"    {tag} ({s.source}) — {s.description}")
            continue

        # Skill 调用: /<skill-name> [args]
        if inp.startswith("/"):
            space_idx = inp.find(" ")
            cmd_name = inp[1:space_idx] if space_idx > 0 else inp[1:]
            cmd_args = inp[space_idx + 1:] if space_idx > 0 else ""
            skill = get_skill_by_name(cmd_name)
            if skill and skill.user_invocable:
                print_info(f"调用 skill: {skill.name}")
                try:
                    if skill.context == "fork":
                        result = execute_skill(skill.name, cmd_args)
                        if result:
                            await agent.chat(f'使用 skill 工具调用 "{skill.name}",参数: {cmd_args or "(无)"}')
                    else:
                        resolved = resolve_skill_prompt(skill, cmd_args)
                        await agent.chat(resolved)
                except Exception as e:
                    if "abort" not in str(e).lower():
                        print_error(str(e))
                continue

        # 普通对话
        try:
            await agent.chat(inp)
        except Exception as e:
            if "abort" not in str(e).lower():
                print_error(str(e))


def main() -> None:
    args = parse_args()

    if args.help:
        print("""
用法: mini-claude [options] [prompt]

Options:
  --yolo, -y          跳过所有确认提示(bypassPermissions 模式)
  --plan              Plan 模式: 只读,描述改动但不执行
  --accept-edits      自动批准文件编辑,但危险 shell 命令仍需确认
  --dont-ask          自动拒绝任何需要确认的操作(适用于 CI)
  --thinking          启用扩展 thinking(仅 Anthropic)
  --model, -m         使用的模型(默认 claude-opus-4-6,或读取 MINI_CLAUDE_MODEL 环境变量)
  --api-base URL      使用 OpenAI 兼容 API 端点(key 通过环境变量提供)
  --resume            恢复上一次会话
  --max-cost USD      预估花费超过该金额时停止
  --max-turns N       运行 N 个 Agent 回合后停止
  --help, -h          显示本帮助

REPL 命令:
  /clear              清空会话历史
  /plan               切换 plan 模式(只读 <-> 普通)
  /cost               显示 token 使用与花费
  /compact            手动压缩会话
  /memory             列出已保存的记忆
  /skills             列出可用 skills
  /<skill-name>       调用一个 skill(例如 /commit "fix types")

示例:
  mini-claude "修复 src/app.ts 里的 bug"
  mini-claude --yolo "跑所有测试并修复失败"
  mini-claude --plan "你会怎么重构这段?"
  mini-claude --max-cost 0.50 --max-turns 20 "实现功能 X"
  OPENAI_API_KEY=sk-xxx mini-claude --api-base https://aihubmix.com/v1 --model gpt-4o "hello"
  mini-claude --resume
  mini-claude  # 启动交互式 REPL
""")
        sys.exit(0)

    permission_mode = _resolve_permission_mode(args)
    model = args.model or os.environ.get("MINI_CLAUDE_MODEL", "claude-opus-4-6")
    api_base = args.api_base

    # 解析 API 配置
    resolved_api_base = api_base
    resolved_api_key: str | None = None
    resolved_use_openai = bool(api_base)

    if os.environ.get("OPENAI_API_KEY") and os.environ.get("OPENAI_BASE_URL"):
        resolved_api_key = os.environ["OPENAI_API_KEY"]
        resolved_api_base = resolved_api_base or os.environ.get("OPENAI_BASE_URL")
        resolved_use_openai = True
    elif os.environ.get("ANTHROPIC_API_KEY"):
        resolved_api_key = os.environ["ANTHROPIC_API_KEY"]
        resolved_api_base = resolved_api_base or os.environ.get("ANTHROPIC_BASE_URL")
        resolved_use_openai = False
    elif os.environ.get("OPENAI_API_KEY"):
        resolved_api_key = os.environ["OPENAI_API_KEY"]
        resolved_api_base = resolved_api_base or os.environ.get("OPENAI_BASE_URL")
        resolved_use_openai = True

    if not resolved_api_key and api_base:
        resolved_api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
        resolved_use_openai = True

    if not resolved_api_key:
        print_error(
            "需要 API key。\n"
            "  Anthropic 格式: 设置 ANTHROPIC_API_KEY(可选 ANTHROPIC_BASE_URL);\n"
            "  OpenAI 兼容格式: 设置 OPENAI_API_KEY + OPENAI_BASE_URL。"
        )
        sys.exit(1)

    agent = Agent(
        permission_mode=permission_mode,
        model=model,
        thinking=args.thinking,
        max_cost_usd=args.max_cost,
        max_turns=args.max_turns,
        api_base=resolved_api_base if resolved_use_openai else None,
        anthropic_base_url=resolved_api_base if not resolved_use_openai else None,
        api_key=resolved_api_key,
    )

    # 恢复会话
    if args.resume:
        session_id = get_latest_session_id()
        if session_id:
            session = load_session(session_id)
            if session:
                agent.restore_session({
                    "anthropicMessages": session.get("anthropicMessages"),
                    "openaiMessages": session.get("openaiMessages"),
                })
            else:
                print_info("没有可恢复的会话。")
        else:
            print_info("没有找到历史会话。")

    prompt = " ".join(args.prompt) if args.prompt else None

    if prompt:
        # 一次性模式
        try:
            asyncio.run(agent.chat(prompt))
        except Exception as e:
            print_error(str(e))
            sys.exit(1)
    else:
        # 交互式 REPL
        asyncio.run(run_repl(agent))


if __name__ == "__main__":
    main()
