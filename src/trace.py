import json

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage


def build_trace(state):
    messages = state["messages"]
    turns = []
    current_turn = None

    for message in messages:
        if isinstance(message, HumanMessage):
            if current_turn is not None:
                turns.append(current_turn)
            current_turn = {
                "user_message": message.content,
                "tool_calls": [],
                "tool_results": [],
                "final_response": None,
            }
        elif isinstance(message, AIMessage):
            if current_turn is None:
                continue
            if message.tool_calls:
                for call in message.tool_calls:
                    current_turn["tool_calls"].append(
                        {"name": call["name"], "args": call["args"]}
                    )
            else:
                current_turn["final_response"] = message.content
        elif isinstance(message, ToolMessage):
            if current_turn is None:
                continue
            current_turn["tool_results"].append(message.content)

    if current_turn is not None:
        turns.append(current_turn)

    return turns


if __name__ == "__main__":
    from src.agent.agent import graph

    kb_config = {"configurable": {"thread_id": "trace-test-kb"}}
    kb_state = graph.invoke(
        {"messages": [("user", "How long do I have to return a bag?")]}, kb_config
    )
    print(json.dumps(build_trace(kb_state), indent=2))

    order_config = {"configurable": {"thread_id": "trace-test-order"}}
    order_state = graph.invoke(
        {"messages": [("user", "Where is my order ORD-1007?")]}, order_config
    )
    print(json.dumps(build_trace(order_state), indent=2))
