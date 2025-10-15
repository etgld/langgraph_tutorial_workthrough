import operator
from functools import partial
from typing import Annotated, TypedDict
from uuid import uuid4

from langchain_core.messages import AnyMessage, SystemMessage, ToolMessage
from langgraph.graph import END, StateGraph

"""
In previous examples we've annotated the `messages` state key
with the default `operator.add` or `+` reducer, which always
appends new messages to the end of the existing messages array.

Now, to support replacing existing messages, we annotate the
`messages` key with a customer reducer function, which replaces
messages with the same `id`, and appends them otherwise.
"""

SOPRANO_SULLIVANT = "You are Tony Soprano, a middle aged Italian American TV character from New Jersey, explain to me what algebraic phylogeny is."


def reduce_messages(
    left: list[AnyMessage], right: list[AnyMessage]
) -> list[AnyMessage]:
    # assign ids to messages that don't have them
    for message in right:
        if not message.id:
            message.id = str(uuid4())
    # merge the new messages with the existing messages
    merged = left.copy()
    for message in right:
        for i, existing in enumerate(merged):
            # replace any existing messages with the same id
            if existing.id == message.id:
                merged[i] = message
                break
        else:
            # append any new messages to the end
            merged.append(message)
    return merged


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], reduce_messages]


class Agent:
    def __init__(self, model, tools, system="", checkpointer=None):
        self.system = system
        graph = StateGraph(AgentState)
        graph.add_node("llm", self.call_openai)
        graph.add_node("action", self.take_action)
        graph.add_conditional_edges(
            "llm", self.exists_action, {True: "action", False: END}
        )
        graph.add_edge("action", "llm")
        graph.set_entry_point("llm")
        self.graph = graph.compile(
            checkpointer=checkpointer, interrupt_before=["action"]
        )
        self.tools = {t.name: t for t in tools}
        self.model = model.bind_tools(tools)

    def call_openai(self, state: AgentState):
        messages = state["messages"]
        if self.system:
            messages = [SystemMessage(content=self.system)] + messages
        message = self.model.invoke(messages)
        return {"messages": [message]}

    def exists_action(self, state: AgentState):
        print(state)
        result = state["messages"][-1]
        return len(result.tool_calls) > 0

    def take_action(self, state: AgentState):
        tool_calls = state["messages"][-1].tool_calls
        print(tool_calls)
        results = []
        print(self.tools)
        for t in tool_calls:
            # tavily_search_results_json
            print(f"Calling: {t}")
            tool = self.tools.get(t["name"])
            if tool is not None:
                result = tool.invoke(t["args"])
            elif tool is None and t["name"] == "tavily_search_results_json":
                tool = self.tools.get("tavily_search")
                if tool is None:
                    result = "NO AVAILABLE TOOL FOUND"
                else:
                    result = tool.invoke(t["args"])
            else:
                result = "NO AVAILABLE TOOL FOUND"
            results.append(
                ToolMessage(tool_call_id=t["id"], name=t["name"], content=str(result))
            )
        print("Back to the model!")
        return {"messages": results}


class GraphExpAgentState(TypedDict):
    lnode: str
    scratch: str
    count: Annotated[int, operator.add]


def node1(state: GraphExpAgentState) -> dict[str, str | int]:
    print(f"node1, count:{state['count']}")
    return {
        "lnode": "node_1",
        "count": 1,
    }


def node2(state: GraphExpAgentState) -> dict[str, str | int]:
    print(f"node2, count:{state['count']}")
    return {
        "lnode": "node_2",
        "count": 1,
    }


def should_continue(limit, state: GraphExpAgentState) -> bool:
    return state["count"] < limit


def graph_builder(limit: int = 3) -> StateGraph:
    builder = StateGraph(AgentState)
    builder.add_node("Node1", node1)
    builder.add_node("Node2", node2)

    builder.add_edge("Node1", "Node2")
    builder.add_conditional_edges(
        "Node2", partial(should_continue, limit), {True: "Node1", False: END}
    )
    builder.set_entry_point("Node1")
    return builder
