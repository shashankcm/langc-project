from dotenv import load_dotenv
from importlib.metadata import version

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic


load_dotenv()

core_version = version("langchain_core")
langgraph_version = version("langgraph")

print(f"langchain-core: {core_version}")
print(f"langgraph: {langgraph_version}")



def main():
    llm_openai = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    response = llm_openai.invoke("Say 'Setup Completed!' in one word")
    print(f"OpenAI Response: {response}")

    llm_anthropic = ChatAnthropic(model_name="claude-haiku-4-5-20251001", temperature=0, timeout=60, stop=None)
    response_anthropic = llm_anthropic.invoke("Say 'Setup Completed!' in one word")
    print(f"Anthropic Response: {response_anthropic}")

    print("Test Setup Complete")


if __name__ == "__main__":
    main()
