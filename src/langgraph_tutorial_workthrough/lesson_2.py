import argparse
import operator
from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI

# from langchain_community.tools.tavily_search import TavilySearchResults
# deprecated per warning
from langchain_tavily import TavilySearch
from langgraph.graph import END, StateGraph

from .api_keys import OPENAI_API_KEY, TAVILY_API_KEY
from .lesson_2_utils import prompt

argparser = argparse.ArgumentParser(description="")

argparser.add_argument(
    "--openai_model",
    type=str,
    default="gpt-3.5-turbo",
)


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    # Don't really like their approach of using
    # typing.Annotated to model this, the other
    # fields of Annotated are generally supposed to model
    # constraints per the documentation https://docs.python.org/3/library/typing.html#typing.Annotated
    # Will try it their way to start but using deque instead of list for the
    # reasons discussed in lesson 1
    # messages: Annotated[deque[AnyMessage], operator.add]


class Agent:
    def __init__(self, model, tools, system=""):
        self.system = system
        graph = StateGraph(AgentState)
        graph.add_node("llm", self.call_openai)
        graph.add_node("action", self.take_action)
        graph.add_conditional_edges(
            "llm", self.exists_action, {True: "action", False: END}
        )
        graph.add_edge("action", "llm")
        graph.set_entry_point("llm")
        self.graph = graph.compile()
        self.tools = {t.name: t for t in tools}
        self.model = model.bind_tools(tools)

    def exists_action(self, state: AgentState):
        result = state["messages"][-1]
        return len(result.tool_calls) > 0

    def call_openai(self, state: AgentState):
        messages = state["messages"]
        if self.system:
            messages = [SystemMessage(content=self.system)] + messages
        message = self.model.invoke(messages)
        # Per the instructional, since messages is built with typing.Annotated we
        # don't have to call it - again, really not my preferred way of expressing this
        return {"messages": [message]}

    def take_action(self, state: AgentState):
        tool_calls = state["messages"][-1].tool_calls
        results = []
        for t in tool_calls:
            print(f"Calling: {t}")
            # if not t["name"] in self.tools:  # check for bad tool name from LLM
            # More idiomatic
            if t["name"] not in self.tools:  # check for bad tool name from LLM
                print("\n ....bad tool name....")
                result = "bad tool name, retry"  # instruct LLM to retry if bad
            else:
                result = self.tools[t["name"]].invoke(t["args"])
            results.append(
                ToolMessage(tool_call_id=t["id"], name=t["name"], content=str(result))
            )
        print("Back to the model!")
        return {"messages": results}


def __process(openai_model: str, system_prompt: str) -> None:
    model = ChatOpenAI(
        api_key=OPENAI_API_KEY, model=openai_model
    )  # reduce inference cost
    tool = TavilySearch(
        tavily_api_key=TAVILY_API_KEY, max_results=4
    )  # increased number of results
    abot = Agent(model, [tool], system=system_prompt)
    # Image(abot.graph.get_graph().draw_png())
    messages = [HumanMessage(content="What is the weather in sf?")]
    result = abot.graph.invoke({"messages": messages})
    print(result["messages"][-1].content)
    messages = [HumanMessage(content="What is the weather in SF and LA?")]
    result = abot.graph.invoke({"messages": messages})
    print(result["messages"][-1].content)
    # Note, the query was modified to produce more consistent results.
    # Results may vary per run and over time as search information and models change.

    query = (
        "Who won the super bowl in 2024? In what state is the winning team headquarters located? \
    What is the GDP of that state? Answer each question."
    )
    messages = [HumanMessage(content=query)]

    model = ChatOpenAI(
        api_key=OPENAI_API_KEY, model="gpt-4o"
    )  # requires more advanced model
    abot = Agent(model, [tool], system=system_prompt)
    result = abot.graph.invoke({"messages": messages})
    print(result["messages"][-1].content)


def main():
    args = argparser.parse_args()
    __process(args.openai_model, prompt)


if __name__ == "__main__":
    main()
