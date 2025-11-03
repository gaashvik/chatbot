from langchain_core.tools import tool
from typing import Dict
from langchain_core.runnables import RunnableConfig
from search.query import SmartSearch
import config


@tool
def retrieve_context(query: str) -> str:
    """retrieve_context, a tool that can be used agents to retrive context in relation to a user query. Accepts a query str in the form of question and outputs the context relavant to that query"""
    retriever = SmartSearch(config.EMBEDDING_MODEL_NAME, config.bedrock_client)
    return retriever._search(query)


@tool
def transition_to_voice(config: RunnableConfig) -> Dict:
    """transition_to_voice, a tool that is used to seemlessely transition between by making an api call to an endpoint with session id for context retrieval. Use this tool if user asks to talk to someone."""
    print(config["configurable"]["thread_id"])
    # api call is made
    thread_id = config["configurable"]["thread_id"]

    return (
        f"Please wait an AI agent will contact you shortly. session id sent:{thread_id}"
    )


tool_list = [retrieve_context, transition_to_voice]
