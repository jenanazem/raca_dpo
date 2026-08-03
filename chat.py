#!/usr/bin/env python3
"""
RACA Legal LLM — Interactive Chat
===================================
Multi-turn chat interface with conversation history.

Usage:
    python chat.py
    python chat.py --compare   # show plain vs steered side by side
"""

import argparse
import json
import sys
import urllib.request
import urllib.error

API_URL = "http://localhost:8080"

def call_api(question: str, history: list, compare: bool = False) -> dict:
    endpoint = "/ask/compare" if compare else "/ask"
    payload = json.dumps({
        "question": question,
        "history": history,
    }).encode("utf-8")
    req = urllib.request.Request(
        API_URL + endpoint,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError:
        print("\n❌ Could not connect to API. Is api.py running?\n")
        sys.exit(1)

def check_health():
    try:
        with urllib.request.urlopen(API_URL + "/health", timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            print(f"✓ Model loaded | {data['hal_features_suppressed']} hallucination features suppressed\n")
    except:
        print("❌ API not reachable. Run: python api.py\n")
        sys.exit(1)

def print_divider():
    print("─" * 60)

def chat_loop(compare: bool):
    print("\n" + "═" * 60)
    print("  RACA Legal LLM — Arabic Legal Assistant")
    print("  اكتب سؤالك القانوني بالعربية")
    print("  Commands: 'exit' to quit | 'clear' to reset history")
    print("═" * 60 + "\n")

    check_health()

    history = []  # list of {"role": "user"/"assistant", "content": "..."}

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye!\n")
            break

        if not question:
            continue

        if question.lower() in ("exit", "quit", "خروج"):
            print("\nGoodbye! / مع السلامة\n")
            break

        if question.lower() == "clear":
            history = []
            print("✓ Conversation history cleared.\n")
            continue

        print("Thinking...", end="\r")
        result = call_api(question, history, compare=compare)

        if compare:
            print_divider()
            print(f"📄 Plain ({result['latency_plain_ms']}ms):")
            print(f"   {result['answer_plain']}")
            print()
            print(f"✨ Steered ({result['latency_steered_ms']}ms):")
            print(f"   {result['answer_steered']}")
            print_divider()
            answer = result['answer_steered']
        else:
            print(f"             ", end="\r")
            print(f"Assistant ({result['latency_ms']}ms):")
            print(f"  {result['answer']}")
            print_divider()
            answer = result['answer']

        # Update history for next turn
        history.append({"role": "user",      "content": question})
        history.append({"role": "assistant",  "content": answer})

        # Keep last 10 turns to avoid context overflow
        if len(history) > 20:
            history = history[-20:]

        print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare", action="store_true", help="Show plain vs steered responses")
    args = parser.parse_args()
    chat_loop(compare=args.compare)
