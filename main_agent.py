from langgraph.graph import StateGraph, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from langchain_aws import ChatBedrock

import argparse
import json
import operator
import math

app = BedrockAgentCoreApp()

# Create price checker tool. We can improve this in the future by calling an API but for now let's make it static.
@tool
def get_btc_price():
    """Fetch the current BTC price in USD"""
    try:
        price = 115000
        return f"Bitcoin price: ${price} on October 27, 2025"
    except Exception as e:
        return f"Failed to fetch BTC price: {e}"

# Define the agent using manual LangGraph construction
def create_agent():
    """Create and configure the LangGraph agent"""
    

    # Initialize your LLM (adjust model and parameters as needed)
    llm = ChatBedrock(
        model_id="us.anthropic.claude-3-7-sonnet-20250219-v1:0",  # or your preferred model
        model_kwargs={"temperature": 0.1}
    )

    # Bind tools to the LLM
    tools = [get_btc_price]
    llm_with_tools = llm.bind_tools(tools)

    # System message
    system_message = "You're a helpful assistant with a extensive background in finance."

    # Define the chatbot node
    def chatbot(state: MessagesState):
        # Add system message if not already present
        messages = state["messages"]
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=system_message)] + messages

        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    # Create the graph
    graph_builder = StateGraph(MessagesState)

    # Add nodes
    graph_builder.add_node("chatbot", chatbot)
    graph_builder.add_node("tools", ToolNode(tools))

    # Add edges
    graph_builder.add_conditional_edges(
        "chatbot",
        tools_condition,
    )
    graph_builder.add_edge("tools", "chatbot")

    # Set entry point
    graph_builder.set_entry_point("chatbot")

    # Compile the graph
    return graph_builder.compile()

# Initialize the agent
agent = create_agent()

@app.entrypoint
def langgraph_bedrock(payload):
    """
    Invoke the agent with a payload
    """
    user_input = payload.get("prompt")

    # Create the input in the format expected by LangGraph
    response = agent.invoke({"messages": [HumanMessage(content=user_input)]})

    # Extract the final message content
    return response["messages"][-1].content

if __name__ == "__main__":
    app.run()