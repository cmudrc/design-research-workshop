"""Chat with the workshop PyBot and inspect which rule produced each answer."""

from __future__ import annotations

import argparse
import json
import sys

from .bot import Content, WorkshopPyBot


def _display(content: Content) -> str:
    """Give participants a compact view of the same JSON returned to agent clients."""
    if "question" in content:
        lines = [content["question"]]
        if "format" in content:
            lines.append(content["format"])
        lines.extend(
            f"  {key} — {value}" for key, value in content.get("tasks", {}).items()
        )
        return "\n".join(lines)
    if "explanation" in content:
        details = content.get("changes") or content.get("evidence")
        return content["explanation"] + (
            "\n" + json.dumps(details, ensure_ascii=False, indent=2) if details else ""
        )
    if "strategy" in content and "concepts" in content:
        lines = [f"{content['brief']} · {content['strategy']} · seed {content['seed']}"]
        for name, row in content.get("chart", {}).items():
            lines.append(f"{name} ({row['source']}) — {', '.join(row['options'])}")
        for i, concept in enumerate(content["concepts"], 1):
            lines.append(
                f"{i}. "
                + " · ".join(f"{axis} — {option}" for axis, option in concept.items())
            )
        return "\n".join(lines)
    if "approved" in content:
        lines = [
            "Critic approved the design."
            if content["approved"]
            else "Critic requested a revision or further evidence."
        ]
        if content.get("feedback"):
            lines.append(content["feedback"])
        lines.extend(content.get("revision_goals", ()))
        return "\n".join(lines)
    if {"feed", "crack", "separate"} <= content.keys():
        return "\n".join(
            f"{field.replace('_', ' ').capitalize()} — {value}"
            for field, value in content.items()
        )
    if "mass_kg" in content and "factor_of_safety" in content:
        members = ", ".join(f"{a}–{b}" for a, b in content["members"])
        return (
            f"{content['objective'].capitalize()} · members {members}\n"
            f"{content['mass_kg']:.2f} kg · safety factor {content['factor_of_safety']:.3f} · "
            f"peak deflection {content['max_deflection_mm']:.3f} mm"
        )
    return json.dumps(content, indent=2, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt", nargs="?", help="Answer one prompt and exit.")
    parser.add_argument(
        "--list", action="store_true", help="List available task rules."
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print each response as a JSON object, including its routing evidence.",
    )
    args = parser.parse_args()
    bot = WorkshopPyBot()
    if args.list:
        for rule in bot.rules:
            print(f"{rule.name} — {rule.description}")
        return

    session = bot.session()
    terminal = args.prompt is None and sys.stdin.isatty()
    if terminal and not args.json:
        print(
            "Workshop PyBot. Enter a design task. Ask ‘Why?’ or ‘What changed?’ after a result. Use /reset or /quit."
        )
    while True:
        try:
            prompt = (
                args.prompt
                if args.prompt is not None
                else input("> " if terminal else "")
            )
        except (EOFError, KeyboardInterrupt):
            if terminal:
                print()
            return
        if prompt == "/quit":
            return
        if prompt == "/reset":
            session.reset()
            if args.prompt is not None:
                return
            continue
        try:
            reply = session.reply(prompt)
        except ValueError as error:
            if args.prompt is not None:
                parser.exit(2, f"{error}\n")
            print(str(error))
            continue
        if args.json:
            print(
                json.dumps(
                    {
                        "rule": reply.rule,
                        "reason": reply.reason,
                        "content": reply.content,
                    },
                    ensure_ascii=False,
                )
            )
        else:
            print(f"[{reply.rule or 'choose a task'} / {reply.reason}]")
            if args.prompt is None:
                print(_display(reply.content))
            else:
                print(json.dumps(reply.content, indent=2, ensure_ascii=False))
        if args.prompt is not None:
            return


if __name__ == "__main__":
    main()
