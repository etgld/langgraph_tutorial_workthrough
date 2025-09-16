# based on https://til.simonwillison.net/llms/python-react-pattern
from collections import deque

from openai import OpenAI

# Not sure why you would ever use .env if you didn't have to...
# from dotenv import load_dotenv
# _ = load_dotenv()
from .lesson_1_utils import action_re, known_actions, prompt
import argparse
argparser = argparse.ArgumentParser(description="")
argparser.add_argument(
    "--openai_api_key",
    type=str,
)

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
            model="gpt-4o", temperature=0, messages=self.messages
        )
        return completion.choices[0].message.content


def __query(client, question, system_prompt, max_turns=5):
    i = 0
    bot = Agent(client=client, system=system_prompt)
    next_prompt = question
    while i < max_turns:
        i += 1
        result = bot(next_prompt)
        print(result)
        actions = [action_re.match(a) for a in result.split("\n") if action_re.match(a)]
        if actions:
            # There is an action to run
            action, action_input = actions[0].groups()
            if action not in known_actions:
                raise Exception(f"Unknown action: {action}: {action_input}")
            print(f" -- running {action} {action_input}")
            observation = known_actions[action](action_input)
            print("Observation:", observation)
            next_prompt = f"Observation: {observation}"
        else:
            return


def __process(openai_api_key: str) -> None:
    question = """I have 2 dogs, a border collie and a scottish terrier. \
    What is their combined weight"""
    client = OpenAI(api_key=openai_api_key)
    __query(client, question, prompt)


def main():
    args = argparser.parse_args()
    __process(args.openai_api_key)


if __name__ == "__main__":
    main()
