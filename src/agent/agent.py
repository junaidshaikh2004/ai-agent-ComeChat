import os 
from dotenv import load_dotenv
import json 

from src.agent.schema import AgentState
from src.agent.prompt import SYSTEM_PROMPT
from src.rag.retriever import retrieve
from src.tools.order_tool import order_lookup_tool
from langchain_groq import ChatGroq
from langchain.tools import tool
from langchain.messages import SystemMessage
from langchain.messages import ToolMessage

from langgraph.graph.state import START,END,StateGraph
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from typing import Literal

load_dotenv()
model = os.getenv("MODEL")
api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    model = model,
    api_key= api_key,
    temperature=0
)

def chatbot(state:AgentState):
    return {
        "messages": [
            model_with_tools.invoke(
                [
                    SystemMessage(
                        content=SYSTEM_PROMPT
                    )
                ]
                + state["messages"]
            )
        ],
    }

#this tool gets the query -> passes it to retriver in the rag , rag gives the retrived docs


@tool
def retriever(query:str):
    """Search Aster & Row's knowledge base for policy and product information relevant to the customer's question."""
    result = retrieve(query , k=6)
    if not result['chunks'] :
        return "No relevant chunk recieved"
    
    parts= []
    for i , (chunk, score) in enumerate(result["chunks"], start=1) :
        doc_id = chunk.metadata.get("document_id", "unknown")
        kb_source = chunk.metadata.get("source","")
        heading = (
        chunk.metadata.get("Header 3")
        or chunk.metadata.get("Header 2")
        or chunk.metadata.get("Header 1")
        or ""
    )
        parts.append(f"[Source {i} | {doc_id} | {heading} | {kb_source} | score: {score:.3f}]\n{chunk.page_content}")
    formatted = "\n\n".join(parts)
    
    if result["conflict"]:
        doc_a, doc_b = result["conflict_pair"]
        formatted += (
            f"\n\nNOTE: {doc_a} and {doc_b} contain conflicting information on this topic. "
            "Do not silently pick one. Tell the customer the information is inconsistent "
            "and recommend human confirmation."
        )
    
    return formatted

STALE_FIELDS_BY_STATUS = {
    "cancelled": {"estimated_delivery", "carrier", "tracking_number"},
    "returned": {"estimated_delivery"},
}

@tool
def order_lookup(order_id: str):
    """Look up an order's status and shipment details using the order ID the customer provides."""
    result = order_lookup_tool(order_id)
    if not result.get("found"):
        return json.dumps(result)

    stale = STALE_FIELDS_BY_STATUS.get(result.get("status"), set())
    clean_result = {k: v for k, v in result.items() if k not in stale}
    return json.dumps(clean_result, indent=2)

tools = [retriever,order_lookup]
tools_by_name = {tool.name: tool for tool in tools}
model_with_tools = llm.bind_tools(tools)
    
def tool_node(state: dict):
    """Performs the tool call"""

    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
    return {"messages": result}

def should_continue(state: AgentState) -> Literal["tool_node", END]:
    """Decide if we should continue the loop or stop based upon whether the LLM made a tool call"""

    messages = state["messages"]
    last_message = messages[-1]

    # If the LLM makes a tool call, then perform an action
    if last_message.tool_calls:
        return "tool_node"

    # Otherwise, we stop (reply to the user)
    return END

# Create the graph
graph_builder = StateGraph(AgentState)

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tool_node", tool_node)

graph_builder.add_edge(START, "chatbot")
graph_builder.add_conditional_edges(
    "chatbot",
    should_continue,
    ["tool_node", END]
)
graph_builder.add_edge("tool_node", "chatbot")

checkpointer = InMemorySaver()
store = InMemoryStore()

graph = graph_builder.compile(checkpointer=checkpointer, store=store)

def save_graph_image(path="graph.png"):
    png_bytes = graph.get_graph().draw_mermaid_png()
    with open(path, "wb") as f:
        f.write(png_bytes)

if __name__ == "__main__":
    save_graph_image()

    config = {"configurable": {"thread_id": "thread-1"}}

    print("--- turn 1 ---")
    for event in graph.stream({"messages": [("user", "My order id is ORD-1001")]}, config):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)

    print("--- turn 2 ---")
    for event in graph.stream({"messages": [("user", "what was my order id?")]}, config):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)