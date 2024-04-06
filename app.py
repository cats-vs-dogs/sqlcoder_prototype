from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.llms import OpenAI
from langchain.agents import AgentType, initialize_agent, AgentExecutor, load_tools
from langchain.chat_models import ChatOpenAI
from langchain.tools import BaseTool, StructuredTool, Tool, tool
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.utilities import SQLDatabase
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate, load_prompt
from langchain_community.utilities import GoogleSearchAPIWrapper 
from tools.financial_tools import RWTool
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from langchain_core.prompts.chat import MessagesPlaceholder
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from pydantic import BaseModel, Field
import math
from scipy.stats import norm
from slimify import *
import re


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
    

def fix_query(query):
    r"""
    Remove postgresql syntax symbols from
    a generated query
    """
    regex = "::\w*"
    fixed = re.sub(regex, "", query)
    return fixed

def generate_sql_query(inp):
    r"""
    Translate a natural language query 
    to sql.
    """
    model_name_or_path = "TheBloke/sqlcoder2-GPTQ"
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                             device_map="auto",
                                             trust_remote_code=False,
                                             revision="gptq-8bit-128g-actorder_True",
                                             )
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    prompt_template = generate_prompt(inp)
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
        top_k=40,
        repetition_penalty=1.1
    )
    sql_pipeline = HuggingFacePipeline(pipeline=pipe)
    out = sql_pipeline(prompt_template)
    out = fix_query(out)
    return out 

def generate_prompt(question, prompt_file="prompt.md", metadata_file="metadata.json"):
    r"""
    Generate a prompt containing the minimal 
    database schema.
    """
    with open(prompt_file, "r") as f:
        prompt = f.read()
    table_metadata_string = generate_slim(metadata_file, question)
    prompt = prompt.format(
        user_question=question, table_metadata_string=table_metadata_string
    )
    return prompt

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

few_shots = { "Give me the total EAD": "SELECT SUM(EAD) FROM Transactions" }
vector_db = initialize_vectorstore(few_shots)
db = SQLDatabase.from_uri("sqlite:///./portfolio_data.db",
                         sample_rows_in_table_info=2)
toolkit = SQLDatabaseToolkit(db=db, llm=OpenAI())
search_tool = load_tools(["google-search"], llm=OpenAI())
search_tool = Tool(
    name="google_search", 
    description="Search Google for recent results.", 
    func=GoogleSearchAPIWrapper().run
)

main_prompt = PromptTemplate.from_template(
"""
    You are an agent designed to interact with a SQL database containing credit risk data.
    Given an input question,  use your tools to create a syntactically correct sqlite
    query to run, then look at the results of the query and return the answer.
    You have access to tools for interacting with the database.
    You have access to the following tools:

    {tools}

    You MUST double check your query before executing it. If you get an error while
    executing a query, rewrite the query and try again.
    Use the following format:

    Question: the input question you must answer

    Thought: you should always think about what to do

    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    Begin!

    Previous conversation history:
    {chat_history}

    Question: {input}

    Thought: I should first get the similar examples I know.
    If the examples are enough to construct the query, I can build it.
    Otherwise, I can then look at the tables in the database to see what I can query.
    Then I should query the schema of the most relevant tables

    Thought: {agent_scratchpad}
""",
)
memory = ConversationBufferWindowMemory(k=4, memory_key="chat_history", return_messages=True)
tools = toolkit.get_tools() + [search_tool, RWTool]
llm=OpenAI(model_name="gpt-4")

agent = create_react_agent(llm, tools, main_prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

#agent = initialize_agent(
#    tools, 
#    llm=OpenAI(model_name="gpt-4"),
#    agent=None,
#    memory=memory,
#    verbose=True,
#    prompt = main_prompt,
#    handle_parsing_errors=True,
#    #agent_kwargs={
#    #    "memory_prompts": [chat_history],
#    #    "input_variables": ["input", "agent_scratchpad", "chat_history"]
#    #},
#)


app = Flask(__name__)
CORS(app)

@app.route('/', methods=['GET', 'POST'])
def index():
   print(memory.buffer_as_str)
   input = request.get_json()["prompt"]
   memory.chat_memory.add_user_message(input)
   out = agent_executor.invoke({
       "input": input,
       "chat_history": memory.chat_memory,
   })
   out = out["output"]
   memory.chat_memory.add_ai_message(input)
   print(out)
   return {'response':out}


#if __name__ == "__main__":
#    memory.chat_memory.add_ai_message("What can I help you with")
#    memory.chat_memory.add_user_message("Say my name")
#    print(main_prompt.format(chat_history=memory.chat_memory, input="NOGGER", agent_scratchpad="N"))