from langchain_core.tools import Tool, tool
from search.query import RAGSearch


@tool
def retrieve_context(query: str) -> str:
    """retrieve_context, a tool that can be used agents to retrive context in relation to a user query. Accepts a query str in the form of question and outputs the context relavant to that query"""
    retiever = RAGSearch()
    return retiever._search(query)


tool_list = [retrieve_context]
