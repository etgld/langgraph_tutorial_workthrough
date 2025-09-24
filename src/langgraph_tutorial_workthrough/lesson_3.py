import json
import re

import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from pygments import formatters, highlight, lexers
from tavily import TavilyClient

from .api_keys import TAVILY_API_KEY

DDG_INST = DDGS()


def search(query, max_results=6, ddg_inst=DDG_INST):
    try:
        results = ddg_inst.text(query, max_results=max_results)
        return [i["href"] for i in results]
    except Exception as e:
        print(
            f"Returning previous results due to following exception reaching ddg - {e}"
        )
        results = [  # cover case where DDG rate limits due to high deeplearning.ai volume
            "https://weather.com/weather/today/l/USCA0987:1:US",
            "https://weather.com/weather/hourbyhour/l/54f9d8baac32496f6b5497b4bf7a277c3e2e6cc5625de69680e6169e7e38e9a8",
        ]
        return results


def scrape_weather_info(url):
    """Scrape content from the given URL"""
    if not url:
        return "Weather information could not be found."

    # fetch data
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        return "Failed to retrieve the webpage."

    # parse result
    soup = BeautifulSoup(response.text, "html.parser")
    return soup


def main() -> None:
    client = TavilyClient(api_key=TAVILY_API_KEY)
    # run search
    print("Basic Searching 'What is in Nvidia's new Blackwell GPU?'")
    result = client.search(
        "What is in Nvidia's new Blackwell GPU?", include_answer=True
    )

    # print the answer
    print(result["answer"])
    # choose location (try to change to your own city!)

    city = "San Francisco"

    query = f"""
        what is the current weather in {city}?
        Should I travel there today?
        "weather.com"
    """

    print("Now doing vanilla duckduckgo search for basic query")

    for i in search(query):
        print(i)
    print("Using DuckDuckGo to find websites and take the first result")
    url = search(query)[0]

    # scrape first wesbsite
    soup = scrape_weather_info(url)

    print(f"Website: {url}\n\n")
    print(str(soup.body)[:50000])  # limit long outputs
    # extract text
    print("Now cleaning up via BeautifulSoup")
    weather_data = [
        tag.get_text(" ", strip=True) for tag in soup.find_all(["h1", "h2", "h3", "p"])
    ]

    # combine all elements into a single string
    weather_data = "\n".join(weather_data)

    # remove all spaces from the combined text
    weather_data = re.sub(r"\s+", " ", weather_data)

    print(f"Website: {url}\n\n")
    print(weather_data)
    print("Now an agentic search using Tavily")
    # run search
    result = client.search(query, max_results=1)

    # print first result
    data = result["results"][0]["content"]

    print("Raw Data")
    print(data)
    # parse JSON
    parsed_json = json.loads(data.replace("'", '"'))

    # pretty print JSON with syntax highlighting
    formatted_json = json.dumps(parsed_json, indent=4)
    colorful_json = highlight(
        formatted_json, lexers.JsonLexer(), formatters.TerminalFormatter()
    )

    print("JSON made to look nicer via pygments")
    print(colorful_json)


if __name__ == "__main__":
    main()
