import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS

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
