"""Chat with the workshop PyBot and inspect which rule produced each answer."""

from __future__ import annotations

import argparse
import json

from .bot import WorkshopPyBot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt", nargs="?", help="Answer one prompt and exit.")
    parser.add_argument(
        "--list", action="store_true", help="List available task rules."
    )
    args = parser.parse_args()
    bot = WorkshopPyBot()
    if args.list:
        for rule in bot.rules:
            print(f"{rule.name} — {rule.description}")
        return

    history = []
    if args.prompt is None:
        print("Workshop PyBot. Enter a design task, /reset, or /quit.")
    while True:
        try:
            prompt = args.prompt if args.prompt is not None else input("> ")
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if prompt == "/quit":
            return
        if prompt == "/reset":
            history.clear()
            if args.prompt is not None:
                return
            continue
        try:
            reply = bot.reply(prompt, history=history)
        except ValueError as error:
            if args.prompt is not None:
                parser.exit(2, f"{error}\n")
            print(str(error))
            continue
        print(f"[{reply.rule or 'choose a task'} / {reply.reason}]")
        print(json.dumps(reply.content, indent=2, ensure_ascii=False))
        if args.prompt is not None:
            return
        history.extend(
            (
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": json.dumps(reply.content)},
            )
        )


if __name__ == "__main__":
    main()
