import os
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import sys
import uuid
import json

from src.agent.agent import graph
from src.trace import build_trace


def main():
    sys.stdout.reconfigure(encoding="utf-8")

    debug = "--debug" in sys.argv

    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    print("Aster & Row Support Agent")
    print("Type 'exit' or 'quit' to stop.")
    print()

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        if not user_input:
            continue

        state = graph.invoke({"messages": [("user", user_input)]}, config)
        answer = state["messages"][-1].content

        print()
        print("Assistant:")
        print(answer)
        print()

        if debug:
            trace = build_trace(state)
            print("--- DEBUG TRACE ---")
            print(json.dumps(trace, indent=2))
            print("--- END DEBUG TRACE ---")
            print()


if __name__ == "__main__":
    main()
