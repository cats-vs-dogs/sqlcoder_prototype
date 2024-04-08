from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser, create_sql_agent, load_tools 
from langchain.prompts import StringPromptTemplate
from langchain.agents import AgentExecutor, load_tools
from langchain.tools import Tool
from langchain.schema import AgentAction, AgentFinish, OutputParserException
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.utilities import SQLDatabase
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate, load_prompt
from langchain_community.utilities import GoogleSearchAPIWrapper 
from tools.financial_tools import RWTool
from flask_cors import CORS
from langchain.agents import AgentExecutor, create_react_agent
from pydantic import BaseModel, Field
from typing import Union
import math
import math, re, os
from scipy.stats import norm
from slimify import *
import uuid
from typing import List, Tuple
from langchain_core.messages.base import BaseMessage


def rw_corp(Input_String: str) -> str:

    """ 
    calculates the risk weight from input parameters 
    PD, LGD, MATURITY, F_LARGE_FIN and SIZE which are fed into the formula in this particular order 
    """    
    PD_s, LGD_s, MATURITY_s, SIZE_s, F_LARGE_FIN_s = Input_String.split(",")

    PD = float(eval(PD_s.split(":")[1].strip()))
    LGD = float(eval(LGD_s.split(":")[1].strip()))
    MATURITY = float(eval(MATURITY_s.split(":")[1].strip()))
    SIZE = float(eval(SIZE_s.split(":")[1].strip()))
    F_LARGE_FIN = (F_LARGE_FIN_s.split(":")[1].strip())
       
    pd_final = max(0.0003, PD)
    size_final = max(5, min(SIZE, 50))    
    r0 = (0.12 * (1.0 - math.exp(-50.0 * pd_final)) / (1.0 - math.exp(-50.0))) + \
        (0.24 * (1.0 - (1 - math.exp(-50.0 * pd_final)) / (1.0 - math.exp(-50.0)))) - \
        (0.04 * (1.0 - (size_final - 5.0) / 45.0))
    if F_LARGE_FIN == 'Y':
        r = r0 * 1.25
    else:
        r = r0
    b = (0.11852 - 0.05478 * math.log(pd_final)) ** 2
    ma = ((1 - 1.5 * b) ** -1) * (1 + (MATURITY - 2.5) * b)    
    rw = ((LGD * norm.cdf((1 - r) ** -0.5 * norm.ppf(pd_final) + (r / (1 - r)) ** 0.5 * norm.ppf(0.999)) - pd_final * LGD) * ma) * 12.5 * 1.06
    return rw  

class RWInput(BaseModel):
    Input_String: str = Field(description='This is a string that contains values for the input parameters \
                              PD, LGD, MATURITY, SIZE and F_LARGE_FIN which are fed into the formula in this particular order ')

def initialize_vectorstore(few_shots={}):
    embeddings = OpenAIEmbeddings()

    few_shot_docs = [Document(page_content=question, metadata={'sql_query': few_shots[question]}) for question in few_shots.keys()]
    return FAISS.from_documents(few_shot_docs, embeddings)

def get_retriever_tool(vector_db):
    retriever = vector_db.as_retriever()
    tool_description = """
    This tool will help you understand similar examples to adapt them to the user question.
    Input to this tool should be the user question.
    IGNORE ITS OUTPUT IF THERE ARENT RELEVANT EXAMPLES.
    """
    return create_retriever_tool(
        retriever,
        name='sql_get_similar_examples',
        description=tool_description
    )

template = """
You are an agent designed to interact with a SQL database.
You are also able to answer other, more general questions as well.
Given an input question, create a syntactically correct sqlite query to run, then look at the results of the query and return the answer.
Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most 10 results.
You can order the results by a relevant column to return the most interesting examples in the database.
Never query for all the columns from a specific table, only ask for the relevant columns given the question.
You have access to tools for interacting with the database.
Only use the below tools. Only use the information returned by the below tools to construct your final answer.
You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

sql_db_query: Input to this tool is a detailed and correct SQL query, output is a result from the database. If the query is not correct, an error message will be returned. If an error is returned, rewrite the query, check the query, and try again. If you encounter an issue with Unknown column 'xxxx' in 'field list', use sql_db_schema to query the correct table fields.
sql_db_schema: Input to this tool is a comma-separated list of tables, output is the schema and sample rows for those tables. Be sure that the tables actually exist by calling sql_db_list_tables first! Example Input: 'table1, table2, table3'
sql_db_list_tables: Input is an empty string, output is a comma separated list of tables in the database.
sql_db_query_checker: Use this tool to double check if your query is correct before executing it. Always use this tool before executing a query with sql_db_query!
calculator: Useful for when you need to answer questions about math.
RWTool: 
    This is a custom tool that calculates the risk weight from a set of input parameters:
        PD - Probability of Default,
        LGD - Loss Given Default,
        MATURITY - Remaining maturity of the loan in years,
        SIZE - The size of the client in MEUR, usually this is the client's turnover, 
        F_LARGE_FIN - If 'Y' the client is a Large Financial Institution   
search_tool: Search Google for recent results.             
            

Use the following format:
Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [sql_db_query, sql_db_schema, sql_db_list_tables, sql_db_query_checker, search_tool, calculator, RWTool]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Previous conversation history:
{history}

Question: {input}
Thought: I should look at the tables in the database to see what I can query. Then I should query the schema of the most relevant tables.
{agent_scratchpad}
"""

class CustomPromptTemplate(StringPromptTemplate):
    template: str
    def format(self, **kwargs) -> str:
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        kwargs["agent_scratchpad"] = thoughts
        return self.template.format(**kwargs)
    

prompt = CustomPromptTemplate(
    template=template,
    input_variables=["input", "intermediate_steps", "history"]
)

class CustomOutputParser(AgentOutputParser):
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        if "Final Answer:" in llm_output:
            return AgentFinish(
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            raise OutputParserException(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)    
output_parser = CustomOutputParser()


class Chatbot():

    db = SQLDatabase.from_uri("sqlite:///./portfolio_data.db",
                         include_tables=['corporate_portfolio'],
                         sample_rows_in_table_info=2
    )

    def __init__(self, memory_window=5):

        llm = ChatOpenAI(model_name='gpt-4',temperature=0)

        toolkit = SQLDatabaseToolkit(db=self.db, llm=llm)
        extra_tools = load_tools(['llm-math'], llm=llm)
        search_tool = load_tools(["google-search"], llm=llm)
        search_tool = Tool(
            name="google_search", 
            description="Search Google for recent results.", 
            func=GoogleSearchAPIWrapper().run
        )
        RWTool = Tool.from_function(
        func=rw_corp,
        name="RWTool",
        description="""
        This is a custom tool that calculates the risk weight from a set of input parameters:
            PD - Probability of Default,
            LGD - Loss Given Default,
            MATURITY - Remaining maturity of the loan in years,
            SIZE - The size of the client in MEUR, usually this is the client's turnover, 
            F_LARGE_FIN - If 'Y' the client is a Large Financial Institution        
            """,
        ) 
      
        extra_tools.append(RWTool)
        extra_tools.append(search_tool)     

        tools = toolkit.get_tools() + list(extra_tools)

        llm_chain = LLMChain(llm=llm, prompt=prompt)
        tool_names = [tool.name for tool in tools]
        agent = LLMSingleActionAgent(
            llm_chain=llm_chain,
            output_parser=output_parser,
            stop=["\nObservation:"],
            allowed_tools=tool_names
        )
        self.memory=ConversationBufferWindowMemory(k=5)
        self.agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True, memory=self.memory)

        # agent = create_react_agent(llm, tools, self.main_prompt)
        # self.agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
        # self.memory = ConversationBufferWindowMemory(k=memory_window, memory_key="chat_history", return_messages=True, handle_parsing_errors=True)

        self.conv_id = uuid.uuid4().int & (1<<31)-1


    def start_conversation(self):
        self.conv_id = uuid.uuid4().int & (1<<31)-1
        self.memory.clear()

    def switch_conversation(self, conv_id: int, messages: List[Tuple[str, str]]):
        self.conv_id = conv_id
        self.memory.clear()
        for entry in messages:
            author, message = entry["author"], entry["message"]
            if author == "AI":
                self.memory.chat_memory.add_ai_message(message)
            elif author == "User":
                self.memory.chat_memory.add_user_message(message)
            else:
                raise Exception(f"Illegal author. {author} could not be recognized")

    def run(self, input: str) -> str:
        """
        """
        self.memory.chat_memory.add_user_message(input)
        # try:
        out = self.agent_executor.invoke({
            "input": input,
            "history": self.memory.chat_memory
        })["output"]
        # except:
            # out = "Sorry, I am unable to answer this question"
        self.memory.chat_memory.add_ai_message(out)
        return out 
