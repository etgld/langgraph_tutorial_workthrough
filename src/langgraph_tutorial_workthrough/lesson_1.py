# based on https://til.simonwillison.net/llms/python-react-pattern
import openai
import re
import httpx
import os
from collections import deque
# Not sure why you would ever use .env if you didn't have to...
# from dotenv import load_dotenv




# _ = load_dotenv()
from openai import OpenAI
class Agent:
    def __init__(self, client, system=""):
        self.system = system
        # changed since list appendage is O(n) (even if under the hood it might be OK)
        # but deque append is contractually O(1)
        self.client = client
        self.messages = deque()
        if self.system:
            self.messages.append({"role": "system", "content": system})

    def __call__(self, message):
        self.messages.append({"role": "user", "content": message})
        result = self.execute()
        self.messages.append({"role": "assistant", "content": result})
        return result

    def execute(self):
        completion = self.client.chat.completions.create(
                        model="gpt-4o", 
                        temperature=0,
                        messages=self.messages)
        return completion.choices[0].message.content
    

def main():
    print("Hello from langgraph-tutorial-workthrough!")


if __name__ == "__main__":
    main()
