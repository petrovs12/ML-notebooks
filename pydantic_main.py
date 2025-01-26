import os
from pydantic_ai import Agent
from dotenv import load_dotenv

load_dotenv()  # Load variables from .env

# Access the OPEN_AI_API environment variable
# OPEN_AI_API = os.getenv('OPEN_AI_API')
openai_api_key = os.getenv("OPENAI_API_KEY")
moo=os.getenv("moo")
print(moo)
# print(openai_api_key)
agent = Agent("openai:gpt-4o")

result_sync = agent.run_sync("What is the capital of Italy?")
print(result_sync.data)
# > Rome


async def main():
    result = await agent.run("What is the capital of France?")
    print(result.data)
    # > Paris

    async with agent.run_stream("What is the capital of the UK?") as response:
        print(await response.get_data())
        # > London
