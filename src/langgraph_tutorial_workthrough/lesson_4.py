import asyncio
import operator
from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph

from .api_keys import OPENAI_API_KEY, TAVILY_API_KEY


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]


class Agent:
    def __init__(self, model, tools, checkpointer, system=""):
        self.system = system
        graph = StateGraph(AgentState)
        graph.add_node("llm", self.call_openai)
        graph.add_node("action", self.take_action)
        graph.add_conditional_edges(
            "llm", self.exists_action, {True: "action", False: END}
        )
        graph.add_edge("action", "llm")
        graph.set_entry_point("llm")
        self.graph = graph.compile(checkpointer=checkpointer)
        self.tools = {t.name: t for t in tools}
        self.model = model.bind_tools(tools)

    def call_openai(self, state: AgentState):
        messages = state["messages"]
        if self.system:
            messages = [SystemMessage(content=self.system)] + messages
        message = self.model.invoke(messages)
        return {"messages": [message]}

    def exists_action(self, state: AgentState):
        result = state["messages"][-1]
        return len(result.tool_calls) > 0

    def take_action(self, state: AgentState):
        tool_calls = state["messages"][-1].tool_calls
        results = []
        for t in tool_calls:
            print(f"Calling: {t}")
            result = self.tools[t["name"]].invoke(t["args"])
            results.append(
                ToolMessage(tool_call_id=t["id"], name=t["name"], content=str(result))
            )
        print("Back to the model!")
        return {"messages": results}


def process(model, tool, prompt) -> None:
    # memory = SqliteSaver.from_conn_string(":memory:")
    with SqliteSaver.from_conn_string(":memory:") as memory:
        abot = Agent(model, [tool], system=prompt, checkpointer=memory)
        messages = [HumanMessage(content="What is the weather in sf?")]

        thread = {"configurable": {"thread_id": "1"}}
        for event in abot.graph.stream({"messages": messages}, thread):
            for v in event.values():
                print(v["messages"])

        messages = [HumanMessage(content="What about in la?")]
        thread = {"configurable": {"thread_id": "1"}}
        for event in abot.graph.stream({"messages": messages}, thread):
            for v in event.values():
                print(v)
        messages = [HumanMessage(content="Which one is warmer?")]
        thread = {"configurable": {"thread_id": "1"}}
        for event in abot.graph.stream({"messages": messages}, thread):
            for v in event.values():
                print(v)
        messages = [HumanMessage(content="Which one is warmer?")]
        thread = {"configurable": {"thread_id": "2"}}
        for event in abot.graph.stream({"messages": messages}, thread):
            for v in event.values():
                print(v)


async def async_process(model, tool, prompt) -> None:
    print("Once more, with async/feeling!")
    async with AsyncSqliteSaver.from_conn_string(":memory:") as memory:
        abot = Agent(model, [tool], system=prompt, checkpointer=memory)
        messages = [HumanMessage(content="What is the weather in SF?")]
        thread = {"configurable": {"thread_id": "4"}}
        async for event in abot.graph.astream_events(
            {"messages": messages}, thread, version="v1"
        ):
            kind = event["event"]
            if kind == "on_chat_model_stream":
                content = event["data"]["chunk"].content
                if content:
                    # Empty content in the context of OpenAI means
                    # that the model is asking for a tool to be invoked.
                    # So we only print non-empty content
                    print(content, end="|")


async def main() -> None:
    tool = TavilySearch(tavily_api_key=TAVILY_API_KEY, max_results=2)

    model = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-4o")

    prompt = """You are a smart research assistant. Use the search engine to look up information. \
    You are allowed to make multiple calls (either together or in sequence). \
    Only look up information when you are sure of what you want. \
    If you need to look up some information before asking a follow up question, you are allowed to do that!
    """
    process(model, tool, prompt)
    await async_process(model, tool, prompt)


if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
