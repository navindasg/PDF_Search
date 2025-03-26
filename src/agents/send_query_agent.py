from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool
from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.tools import query_db
from agents.get_db_info_agent import get_db_info_agent

def send_query_agent(user_query="",model="gemma3:27b", db_info="", debug=False):
    llm = OllamaLLM(
        model=model,
        #base_url="http://localhost:11434",
        #max_tokens=1024,
        #temperature=0.0
    )
    template = f"""Here is what the user wants: {user_query}. 
    Here is some information about the database you should reference to complete the users request: {db_info}
    Please accomplish the task by sending a query to the database.

    The database is a Microsoft SQL Server database.
    Use a query that should work on any Microsoft SQL Server database.

    ***YOUR QUERIES SHOULD NOT HAVE ANY FORMATTING SUCH AS NEW LINES, SINGLE OR DOUBLE QUOTES TO DENOTE A STRING OR ANYTHING EXCEPT THE QUERY.***

    ***If an error occurs please try again with a different query.
    ***All queries should be one line.
    ***If you need more than one line use multiple queries.
    ***Do not assume anything about the database except the following:
    1. There is a databse called table_db
    2. Inside of table_db there is a schema called public
    3. Inside of public there are tables. All of the relevant information will be in these tables."""

    prompt_template = PromptTemplate(
        input_variables=["user_query","db_info"],
        template=template,
    )

    tools_for_agent = [
        Tool(
            name="query_db",
            func=query_db,
            description="Query the database"
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
    
    formatted_prompt = prompt_template.format(user_query=user_query, db_info=db_info)
    results = agent_executor.invoke({"input": formatted_prompt})

    if debug:
        print(results["output"])
    return results["output"]


if __name__ == "__main__":
    db_info = get_db_info_agent(debug=True)
    print(send_query_agent(user_query="how many rows are in each table?", db_info=db_info, debug=True))