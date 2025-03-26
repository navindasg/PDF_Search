from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool
from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.tools import similarity_search

def find_context_agent(user_query="",model="gemma3:27b", debug=False):
    llm = OllamaLLM(
        model=model,
        #base_url="http://localhost:11434",
        #max_tokens=1024,
        #temperature=0.0
    )

    template = f"""Here is the user query: {user_query}
    Please find the most relevant information to the user query from the vector store.
    """

    prompt_template = PromptTemplate(
        input_variables=["user_query"],
        template=template,
    )
    
    tools_for_agent = [
        Tool(
            name="similarity_search",
            func=similarity_search,
            description="Search the vector store for the most relevant information to the user query",
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
    print(find_context_agent(user_query="what is the price of eggs?", debug=True))