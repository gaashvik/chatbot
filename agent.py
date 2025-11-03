from langgraph.graph import StateGraph, MessagesState, START, END
from typing import Annotated, Dict, List, Literal
from langgraph.checkpoint.mongodb import MongoDBSaver
from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage
import config
from langchain_aws import ChatBedrockConverse
from tools import tool_list
import prompt


class ChatBot:
    def __init__(self, bedrock_client=None):

        self.llm = ChatBedrockConverse(
            model_id="anthropic.claude-3-haiku-20240307-v1:0", client=bedrock_client
        )
        self.llm_with_tools = self.llm.bind_tools(tool_list)
        self.app = self._build_graph()
        self.tools_by_name = {tool.name: tool for tool in tool_list}
        self.system_prompt = prompt.system_prompt

    def _build_graph(self):
        checkpointer = MongoDBSaver(config.mongo_client, db_name="aivar")

        graph = StateGraph(MessagesState)

        graph.add_node("agent", self._answer_node)
        graph.add_node("tools", self._tool_node)
        # graph.add_node("continue",self._continue_node)

        graph.add_edge(START, "agent")
        graph.add_conditional_edges(
            "agent", self._continue_node, {"Action": "tools", END: END}
        )
        graph.add_conditional_edges(
            "tools", self._tool_decision_node, {"Other": "agent", END: END}
        )
        # graph.add_edge("agent", END)

        return graph.compile()

    def _answer_node(self, state: MessagesState) -> Dict[str, List]:
        """Agent Node responsible for processing user query and responds accordingly"""
        system_message = [SystemMessage(content=self.system_prompt)]
        messages = state["messages"]
        result = self.llm_with_tools.invoke(system_message + messages)

        return {"messages": [result]}

    def _continue_node(self, state: MessagesState) -> Literal["Action", END]:
        """continue node responsible for routing state to appropriate node based on tool call request or response"""

        messages = state["messages"]
        last_message = messages[-1]

        if last_message.tool_calls:
            return "Action"

        return END

    def _tool_decision_node(self, state: MessagesState) -> Literal["Other", END]:
        """tool_descison_node responsible for wither routing back to agent or ending the conversation."""
        messages = state["messages"]
        last_message = messages[-1]
        print("[TOOOL CALLL]", last_message.kwargs)
        if last_message.kwargs["tool_call_name"] == "transition_to_voice":
            print("Intiating transition.....")
            return END

        return "Other"

    def _tool_node(self, state: MessagesState) -> Dict[str, List]:
        """tool node contains a list of tools available to the agent"""

        result = []
        messages = state["messages"]
        last_message = messages[-1]

        for tool_call in last_message.tool_calls:
            tool = self.tools_by_name[tool_call["name"]]
            observation = tool.invoke(tool_call["args"])
            result.append(
                ToolMessage(
                    content=observation,
                    tool_call_id=tool_call["id"],
                    kwargs={"tool_call_name": tool_call["name"]},
                )
            )
        return {"messages": result}

    def execute(self, user_input: str, thread_id: str) -> str:
        """Execute the graph with user input"""

        messages = [HumanMessage(content=user_input)]
        input_state = {"messages": messages}
        thread_config = {"configurable": {"thread_id": thread_id}}

        outputs = list(self.app.stream(input_state, thread_config))
        print(self.app.get_state(thread_config))
