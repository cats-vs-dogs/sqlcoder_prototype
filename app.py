from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.llms import OpenAI
from langchain.agents import AgentType, initialize_agent, AgentExecutor
from langchain.chains import LLMMathChain
from langchain.chat_models import ChatOpenAI
from langchain.tools import BaseTool, StructuredTool, Tool, tool
from langchain.utilities import SerpAPIWrapper
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.utilities import SQLDatabase
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain.agents.agent import AgentExecutor
import mlflow
import pandas as pd
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain
from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from slimify import *


def generate_sql_query(inp):
    model_name_or_path = "TheBloke/sqlcoder2-GPTQ"
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                             device_map="auto",
                                             trust_remote_code=False,
                                             revision="main")
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
    return out 




def generate_prompt(question, prompt_file="prompt.md", metadata_file="metadata.json"):
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

few_shots = {
    "Give me the total EAD" : "SELECT SUM(EAD) FROM Transactions"
}
vector_db = initialize_vectorstore(few_shots)

db = SQLDatabase.from_uri("sqlite:///./portfolio.db",
                         sample_rows_in_table_info=2
)
toolkit = SQLDatabaseToolkit(db=db, llm=OpenAI())


tools = [
    Tool.from_function(
        func=generate_sql_query,
        name="Generate SQL",
        description="Input to this tool is natural language that needs to be converted to SQL query. INPUT IS NOT AN SQL QUERY! Output is a correct SQL query."
    ),
    get_retriever_tool(vector_db)
]
tools = tools+toolkit.get_tools();

memory = ConversationBufferWindowMemory(k=4, memory_key="history")

custom_suffix = """
Thought:
I should first get the similar examples I know.
If the examples are enough to construct the query, I can build it.
Otherwise, I can then look at the tables in the database to see what I can query.
Then I should query the schema of the most relevant tables
"""
prompt_temp = """
You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct sqlite query to run, then look at the results of the query and return the answer.
Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most 10 results.
You can order the results by a relevant column to return the most interesting examples in the database.
Never query for all the columns from a specific table, only ask for the relevant columns given the question.
You have access to tools for interacting with the database.
You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.


DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.


If the question does not seem related to the database, YOU MUST RETURN ###### as the answer.

Use the following format:
Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [sql_db_query, sql_db_schema, sql_db_list_tables, sql_db_query_checker, Calculator, RWTool, sql_get_similar_examples]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Previous conversation history:
{history}

Question: {input}
Thought: I should first get the similar examples I know.
If the examples are enough to construct the query, I can build it.
Otherwise, I can then look at the tables in the database to see what I can query.
Then I should query the schema of the most relevant tables
{agent_scratchpad}
"""

orca_template = f"""
### Instruction:

{prompt_temp}

### Response:
"""

#llm = LlamaCpp(
#    model_path="./ggml-model-f16.gguf",
#    temperature=0.75,
#    max_tokens=2000,
#    top_p=1,
#    n_ctx=2048,
#    verbose=True,  # Verbose is required to pass to the callback manager
#)


agent = initialize_agent(
    tools, 
    llm=OpenAI(),
    agent=None, #AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    verbose=True,
    prompt = PromptTemplate.from_template(orca_template),
    handle_parsing_errors=True,
)

#agent.agent.llm_chain.prompt.template = agent.agent.llm_chain.prompt.template.format(chat_history="{chat_history}", input="{input}", agent_scratchpad = custom_suffix+ "{agent_scratchpad} ")

app = Flask(__name__)
CORS(app)

@app.route('/', methods=['GET', 'POST'])
def index():
    prompt = request.get_json()["prompt"]
    out = agent.run(prompt)
    return {'response':out}

if __name__ == '__main__':
    #print(agent.run("What is the date with the smallest EAD?"))
    print(generate_sql_query("What is the user with the lowest LGD?"))
    #app.run(debug=True)
