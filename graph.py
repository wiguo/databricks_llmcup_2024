# Databricks notebook source
import os
import getpass
import pandas as pd
from sqlalchemy import create_engine
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_core.tools import Tool, BaseTool
from langchain_experimental.utilities import PythonREPL
from typing import Optional, Any
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
import matplotlib.pyplot as plt
from langchain_community.tools import YouTubeSearchTool
from langchain_databricks import ChatDatabricks
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_core.tools import Tool, BaseTool
from langchain_core.output_parsers import StrOutputParser
from langchain_databricks import ChatDatabricks
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_core.tools import Tool, BaseTool
from langchain.agents import AgentType, initialize_agent,create_structured_chat_agent, create_react_agent, AgentExecutor
import mlflow
from typing import TypedDict, Annotated, Sequence
import operator
from langchain_core.messages import BaseMessage, FunctionMessage, HumanMessage, ToolMessage, SystemMessage
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages, AnyMessage
from langgraph.graph import StateGraph, END
from IPython.display import Image, display
from langgraph.graph.state import CompiledStateGraph

# COMMAND ----------

# DBTITLE 1,SQL TOOL

def load_graph() -> CompiledStateGraph:
    df = pd.read_csv("hf://datasets/aibabyshark/insurance_customer_support_QA_result/insurance_customer_support_conversation_with_qa_result.csv")
    engine = create_engine("sqlite:///qa.db")
    df.to_sql("qa_table",engine, if_exists='replace', index=False)
    db = SQLDatabase.from_uri("sqlite:///qa.db")

    from langchain.chat_models import ChatDatabricks

    chat_model = ChatDatabricks(endpoint="databricks-meta-llama-3-1-405b-instruct", max_tokens=4000)

    sql_agent = create_sql_agent(chat_model, db=db, agent_type="zero-shot-react-description", verbose=False)


    sql_tool = Tool.from_function(
        name="sql_agent", 
        func=sql_agent.run, 
        description="useful for answering questions by using SQL",
        handle_tool_error = True,
        verbose = False, 
        error_message = "Error running sql code",
    ) 

    class PythonREPLTool(BaseTool):
        # ...

        def _run(
            self,
            query: str,
            run_manager: Optional[CallbackManagerForToolRun] = None,
        ) -> Any:
            """Use the tool."""
            if self.sanitize_input:
                query = sanitize_input(query)
            result = self.python_repl.run(query)

            # Check if the result is a matplotlib figure
            if isinstance(result, plt.Figure):
                # Save the figure to a file
                result.savefig('output.png')

            return result
        

    python_tool = Tool.from_function(
        name="python_tool",
        func=PythonREPL().run,
        description="run python code to save charts for sql results in current working directory, do not open it",
        handle_tool_error = True,
        verbose = False, 
        error_message = "Error running python code",
    )

    youtube_tool = YouTubeSearchTool()




    llm = ChatDatabricks(endpoint="databricks-meta-llama-3-1-70b-instruct", max_tokens=4000)
    feedback_prompt = PromptTemplate(
        template="""Given the following question {input}.
        Provide training feedbacks for the mentioned customer support agent. 
        You must ask sql_agent to get the relevant qa_feedback_summary column from qa_table for the mentioned customer support agent and then summarize it. 
        Write a detailed response based on the qa_feedback_summary column from qa_table only. 
        You should NOT generate your own data.
        You should NOT assume any data or returned data.
        You can end with a training plan for the customer support agent to improve the weakest area. 
        When confronted with choices, make a decision yourself with reasoning.
        """,
        input_variables=["input"],
    )


    llm_chain = LLMChain(llm=llm, prompt=feedback_prompt, verbose=False)

    feedback_tool = Tool(
        name="feedback_agent",
        func=llm_chain.run,
        description="useful for providing training feedbacks based on qa_feedback_summary from sql_agent",
        handle_tool_error = True,
        verbose = False, 
        error_message = "Error running feedback tool",
    )
   
    os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI Token: ")
    main_llm =  ChatOpenAI(model="gpt-4o")
    # Alternative main llm
    # main_llm = ChatDatabricks(endpoint="databricks-meta-llama-3-1-70b-instruct", max_tokens=4000)
    
    tools = [sql_tool, python_tool, feedback_tool, youtube_tool]
    llm_with_tools = main_llm.bind_tools(tools)

    class AgentState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], operator.add]

    def call_model(state:AgentState):
        messages = state["messages"]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    def should_continue(state:AgentState):
        messages = state["messages"]
        last_message = messages[-1]

        if not last_message.tool_calls:
            return "end"
        else:
            return "continue"
        
    tool_node = ToolNode(tools)

    workflow = StateGraph(AgentState)

    workflow.add_node("agent",call_model)
    workflow.add_node("tools", tool_node)

    workflow.set_entry_point("agent")

    workflow.add_conditional_edges("agent", 
                                    should_continue,
                                    {
                                        "continue": "tools",
                                        "end": END
                                    }
                                    )
    workflow.add_edge("tools", "agent")
    new_app = workflow.compile()
    return new_app

# Set are model to be leveraged via model from code
mlflow.models.set_model(load_graph())

