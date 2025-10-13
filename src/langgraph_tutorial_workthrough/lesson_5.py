from langchain_core.messages import HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langgraph.checkpoint.sqlite import SqliteSaver
from .api_keys import TAVILY_API_KEY, OPENAI_API_KEY
from .lesson_5_utils import Agent


def thread_and_state_change_experiments() -> None:
    tool = TavilySearch(tavily_api_key=TAVILY_API_KEY, max_results=2)
    prompt = """You are a smart research assistant. Use the search engine to look up information. \
    You are allowed to make multiple calls (either together or in sequence). \
    Only look up information when you are sure of what you want. \
    If you need to look up some information before asking a follow up question, you are allowed to do that!
    """
    model = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-3.5-turbo")
    sf_query_messages = [HumanMessage(content="Whats the weather in SF?")]
    sf_query_thread = {"configurable": {"thread_id": "1"}}
    la_query_messages = [HumanMessage("Whats the weather in LA?")]
    la_query_thread = {"configurable": {"thread_id": "2"}}
    state_mod_la_query_messages = [HumanMessage("Whats the weather in LA?")]
    state_mod_la_query_thread = {"configurable": {"thread_id": "3"}}
    with SqliteSaver.from_conn_string(":memory:") as memory:
        abot = Agent(model, [tool], system=prompt, checkpointer=memory)
        for event in abot.graph.stream(
            {"messages": sf_query_messages}, sf_query_thread
        ):
            for v in event.values():
                print(v)
        abot.graph.get_state(sf_query_thread)
        abot.graph.get_state(sf_query_thread).next
        for event in abot.graph.stream(None, sf_query_thread):
            for v in event.values():
                print(v)
        abot.graph.get_state(sf_query_thread)
        abot.graph.get_state(sf_query_thread).next
        for event in abot.graph.stream(
            {"messages": la_query_messages}, la_query_thread
        ):
            for v in event.values():
                print(v)
        while abot.graph.get_state(la_query_thread).next:
            print("\n", abot.graph.get_state(la_query_thread), "\n")
            _input = input("proceed?")
            if _input != "y":
                print("aborting")
                break
            for event in abot.graph.stream(None, la_query_thread):
                for v in event.values():
                    print(v)

        for event in abot.graph.stream(
            {"messages": state_mod_la_query_messages}, state_mod_la_query_thread
        ):
            for v in event.values():
                print(v)
        current_values = abot.graph.get_state(state_mod_la_query_thread)
        # Wouldn't it be hilarious if someone
        # implemented "call with current continuation" for
        # this stuff?
        _id = current_values.values["messages"][-1].tool_calls[0]["id"]
        current_values.values["messages"][-1].tool_calls = [
            {
                "name": "tavily_search_results_json",
                # Get it? The _other_ LA....
                # ..............
                # ..............
                # ..............
                # ..............
                "args": {"query": "current weather in Louisiana"},
                "id": _id,
            }
        ]
        abot.graph.update_state(state_mod_la_query_thread, current_values.values)
        abot.graph.get_state(state_mod_la_query_thread)
        for event in abot.graph.stream(None, state_mod_la_query_thread):
            for v in event.values():
                print(v)
        states = []
        for state in abot.graph.get_state_history(state_mod_la_query_thread):
            print(state)
            print("--")
            states.append(state)
        to_replay = states[-3]
        for event in abot.graph.stream(None, to_replay.config):
            for v in event.values():
                print(v)
        _id = to_replay.values["messages"][-1].tool_calls[0]["id"]
        to_replay.values["messages"][-1].tool_calls = [
            {
                "name": "tavily_search_results_json",
                "args": {"query": "current weather in LA, accuweather"},
                "id": _id,
            }
        ]
        branch_state = abot.graph.update_state(to_replay.config, to_replay.values)
        for event in abot.graph.stream(None, branch_state):
            for k, v in event.items():
                if k != "__end__":
                    print(v)
        _id = to_replay.values["messages"][-1].tool_calls[0]["id"]
        state_update = {
            "messages": [
                ToolMessage(
                    tool_call_id=_id,
                    name="tavily_search_results_json",
                    content="54 degree celcius",
                )
            ]
        }
        branch_and_add = abot.graph.update_state(
            to_replay.config, state_update, as_node="action"
        )

        for event in abot.graph.stream(None, branch_and_add):
            for k, v in event.items():
                print(v)


def time_travel() -> None:
    pass


def main() -> None:
    thread_and_state_change_experiments()


if __name__ == "__main__":
    main()
