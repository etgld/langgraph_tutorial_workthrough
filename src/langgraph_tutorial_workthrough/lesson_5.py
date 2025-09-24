from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langgraph.checkpoint.sqlite import SqliteSaver
from .api_keys import TAVILY_API_KEY, OPENAI_API_KEY
from .lesson_5_utils import Agent


def main() -> None:
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
    with SqliteSaver.from_conn_string(":memory:") as memory:
        abot = Agent(model, [tool], system=prompt, checkpointer=memory)
        for event in abot.graph.stream({"messages": sf_query_messages}, sf_query_thread):
            for v in event.values():
                print(v)
        abot.graph.get_state(sf_query_thread)
        abot.graph.get_state(sf_query_thread).next
        for event in abot.graph.stream(None, sf_query_thread):
            for v in event.values():
                print(v)
        abot.graph.get_state(sf_query_thread)
        abot.graph.get_state(sf_query_thread).next
        for event in abot.graph.stream({"messages": la_query_messages}, la_query_thread):
            for v in event.values():
                print(v)
        while abot.graph.get_state(la_query_thread).next:
            print("\n", abot.graph.get_state(la_query_thread),"\n")
            _input = input("proceed?")
            if _input != "y":
                print("aborting")
                break
            for event in abot.graph.stream(None, la_query_thread):
                for v in event.values():
                    print(v)

if __name__ == "__main__":
    main()
