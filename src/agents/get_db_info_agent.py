from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool
from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.tools import list_tables_and_columns, query_db

def get_db_info_agent(model="gemma3:27b", debug=False):
    llm = OllamaLLM(
        model=model,
        #base_url="http://localhost:11434",
        #max_tokens=1024,
        #temperature=0.0

    )
    template = """Find information such as database names, schema names, table names, and column names about the database by sending it a query.
    The database is a Microsoft SQL Server database. Use a query that should work on any Postgresql database.

    ***YOUR QUERIES SHOULD NOT HAVE ANY FORMATTING SUCH AS NEW LINES, SINGLE OR DOUBLE QUOTES TO DENOTE A STRING OR ANYTHING EXCEPT THE QUERY.***

    ***If an error occurs please try again with a different query.
    ***All queries should be one line.
    ***Do not assume anything about the database schema except the following:
    1. There is a databse called table_db
    2. Inside of table_db there is a schema called public
    3. Inside of public there are tables. All of the relevant information will be in these tables."""
    prompt_template = PromptTemplate(
        template=template,
    )

    tools_for_agent = [
        Tool(
            name="list_tables_and_columns",
            func=list_tables_and_columns,
            description="List all relevanttables and columns in the database"
        ),
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
        verbose=debug
    )

    results = agent_executor.invoke(
        {"input": prompt_template.format()}
    )
    if debug:
        print(results["output"])
    return results["output"]


if __name__ == "__main__":
    print(get_db_info_agent(debug=True))