from typing import Any, Dict

import boto3
from langchain import hub
from langchain.agents import create_react_agent, tool
from langchain.schema import AgentFinish
from langchain.tools import tool
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.llms import Bedrock
from langchain_community.vectorstores import FAISS
from langchain_core.agents import AgentFinish
from langchain_core.runnables import RunnablePassthrough
from langgraph.graph import END, Graph

# Setup bedrock
bedrock_runtime = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-east-1",
)

# Define Bedrock LLM
LLM = Bedrock(client=bedrock_runtime, model_id="anthropic.claude-v2")
LLM.model_kwargs = {"temperature": 0.7, "max_tokens_to_sample": 4096}

@tool
def well_arch_tool(query: str) -> Dict[str, Any]:
    """Returns text from AWS Well-Architected Framework releated to the query"""
    embeddings = BedrockEmbeddings(
        client=bedrock_runtime,
        model_id="amazon.titan-embed-text-v1",
    )
    vectorstore = FAISS.load_local("local_index", embeddings)
    docs = vectorstore.similarity_search(query)

    resp_json = {"docs": docs}

    return resp_json


TOOLS = [well_arch_tool]

def construct_agent():
    # Get the prompt to use - you can modify this!
    prompt = hub.pull("hwchase17/react")
    # Adding a custom header
    prompt.template = """You are an expert AWS Certified Solutions Architect. Your role is to help customers understand best practices on building on AWS. You will always reference the AWS Well-Architected Framework when customers ask questions on building on AWS. """ + prompt.template 
    # print(prompt.template)
    return create_react_agent(LLM, TOOLS, prompt)


def create_graph_workflow(agent_runnable):
    agent = RunnablePassthrough.assign(agent_outcome = agent_runnable)
    workflow = Graph()
    # Add the agent node, we give it name `agent` which we will use later
    workflow.add_node("agent", agent)
    # Add the tools node, we give it name `tools` which we will use later
    workflow.add_node("tools", execute_tools)


    # Set the entrypoint as `agent`
    # This means that this node is the first one called
    workflow.set_entry_point("agent")

    # We now add a conditional edge
    workflow.add_conditional_edges(
        # First, we define the start node. We use `agent`.
        # This means these are the edges taken after the `agent` node is called.
        "agent",
        # Next, we pass in the function that will determine which node is called next.
        should_continue,
        # Finally we pass in a mapping.
        # The keys are strings, and the values are other nodes.
        # END is a special node marking that the graph should finish.
        # What will happen is we will call `should_continue`, and then the output of that
        # will be matched against the keys in this mapping.
        # Based on which one it matches, that node will then be called.
        {
            # If `tools`, then we call the tool node.
            "continue": "tools",
            # Otherwise we finish.
            "exit": END
        }
    )

    # We now add a normal edge from `tools` to `agent`.
    # This means that after `tools` is called, `agent` node is called next.
    workflow.add_edge('tools', 'agent')

    # Finally, we compile it!
    # This compiles it into a LangChain Runnable,
    # meaning you can use it as you would any other runnable
    return workflow.compile()


# Define the function to execute tools
def execute_tools(data):
    # Get the most recent agent_outcome - this is the key added in the `agent` above
    agent_action = data.pop('agent_outcome')
    # Get the tool to use
    tool_to_use = {t.name: t for t in TOOLS}[agent_action.tool]
    # Call that tool on the input
    observation = tool_to_use.invoke(agent_action.tool_input)
    # We now add in the action and the observation to the `intermediate_steps` list
    # This is the list of all previous actions taken and their output
    data['intermediate_steps'].append((agent_action, observation))
    return data


# Define logic that will be used to determine which conditional edge to go down
def should_continue(data):
    # If the agent outcome is an AgentFinish, then we return `exit` string
    # This will be used when setting up the graph to define the flow
    if isinstance(data['agent_outcome'], AgentFinish):
        return "exit"
    # Otherwise, an AgentAction is returned
    # Here we return `continue` string
    # This will be used when setting up the graph to define the flow
    else:
        return "continue"
    
def main():
    # Create Agent
    agent_runnable = construct_agent()
    # Create LangGraph Workflow with Agent as Entrypoint
    chain = create_graph_workflow(agent_runnable)
    # Invoke the LangGraph Workflow with input and intermediate steps
    result = chain.invoke({"input": "What does the AWS Well-Architected Framework say about how to create secure VPCs?", "intermediate_steps": []})
    # Print the output of the LangGraph Workflow - this is the output of the Agent
    output = result['agent_outcome'].return_values["output"]
    print(output)

if __name__ == "__main__":
    main()
     
