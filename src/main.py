import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool
from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.get_db_info_agent import get_db_info_agent
from agents.find_context_agent import find_context_agent
from agents.send_query_agent import send_query_agent

def main(user_query="", debug=False):
    llm = OllamaLLM(
        model="gemma3:27b",
        base_url="http://localhost:11434",
        max_tokens=1024,
        temperature=0.0
    )

    template = """
    You are a helpful assistant what will use the tools provided to answer the user's question.

    Here is the user's question: {user_query}

    The information will either be found from the context provided or from the database.
    You can get the context by using the "find_context_agent" tool.
    You can get the database information by using the "get_db_info_agent" tool.
    You can send a query to the database by using the "send_query_agent" tool.
    Please search all possible sources for the information.
    Use the database information to write an accurate query.
     ***YOUR QUERIES SHOULD NOT HAVE ANY FORMATTING SUCH AS NEW LINES, SINGLE OR DOUBLE QUOTES TO DENOTE A STRING OR ANYTHING EXCEPT THE QUERY.***

    ***If an error occurs please try again with a different query.
    ***All queries should be one line.
    ***If you need more than one line use multiple queries.
    ***Do not assume anything about the database except the following:
    1. There is a databse called table_db
    2. Inside of table_db there is a schema called public
    3. Inside of public there are tables. All of the relevant information will be in these tables
    """

    prompt_template = PromptTemplate(
        input_variables=["user_query"],
        template=template,
    )

    tools_for_agent = [
        Tool(
            name="find_context_agent",
            func=find_context_agent,
            description="Find the context from a vector store for the user's question"
        ),
        Tool(
            name="get_db_info_agent",
            func=get_db_info_agent,
            description="Get information about the database"
        ),
        Tool(
            name="send_query_agent",
            func=send_query_agent,
            description="Send a query to the database"
        )
    ]

    react_prompt = hub.pull("hwchase17/react")

    agent = create_react_agent(
        llm=llm,
        tools=tools_for_agent,
        prompt=react_prompt,
    )

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools_for_agent,
        verbose=debug,
        handle_parse_errors=True
    )

    formatted_prompt = prompt_template.format(user_query=user_query)
    results = agent_executor.invoke({"input": formatted_prompt})

    if debug:
        print(results["output"])
    return results["output"]

if __name__ == "__main__":
    main(user_query="What is the price of eggs?", debug=True)