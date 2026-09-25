"""
Quick terminal test harness.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python cli.py
"""

from bot import SupportBot


def main() -> None:
    bot = SupportBot()
    print("Support bot ready. Type 'quit' to exit, 'reset' to clear history.\n")

    while True:
        user_message = input("You: ").strip()
        if not user_message:
            continue
        if user_message.lower() == "quit":
            break
        if user_message.lower() == "reset":
            bot.reset()
            print("(conversation cleared)\n")
            continue

        result = bot.send(user_message)
        print(f"\nBot: {result['reply']}\n")

        if result["sources"]:
            print(f"[used articles: {', '.join(result['sources'])}]")
        if result["escalate"]:
            print("[flagged for human escalation]")
        print()


if __name__ == "__main__":
    main()
