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
from langchain.agents.agent import AgentExecutor
import mlflow
import pandas as pd
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS


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
    return sql_pipeline(prompt_template)




def generate_prompt(question, prompt_file="prompt.md", metadata_file="metadata.sql"):
    with open(prompt_file, "r") as f:
        prompt = f.read()
    
    with open(metadata_file, "r") as f:
        table_metadata_string = f.read()

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
    """

    return create_retriever_tool(
        retriever,
        name='sql_get_similar_examples',
        description=tool_description
    )

few_shots = {
    "Give me the total exposure undrawn for 2022-12-31" : "SELECT SUM(EXPOSURE_UNDRAWN) FROM corporate_portfolio WHERE REPORTING_DATE = 2022-12-31"
}
vector_db = initialize_vectorstore(few_shots)

db = SQLDatabase.from_uri("sqlite:///./portfolio_data.db",
                         include_tables=['corporate_portfolio'],
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

memory = ConversationBufferWindowMemory(k=4, memory_key="chat_history")

custom_suffix = """
Thought:
I should first get the similar examples I know.
If the examples are enough to construct the query, I can build it.
Otherwise, I can then look at the tables in the database to see what I can query.
Then I should query the schema of the most relevant tables
"""

agent = initialize_agent(
    tools, 
    llm=OpenAI(),
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    verbose=True,
    handle_parsing_errors=True,
)

agent.agent.llm_chain.prompt.template = agent.agent.llm_chain.prompt.template.format(chat_history="{chat_history}", input="{input}", agent_scratchpad = custom_suffix+ "{agent_scratchpad} ")

app = Flask(__name__)
CORS(app)

@app.route('/', methods=['GET', 'POST'])
def index():
    prompt = request.get_json()["prompt"]
    out = agent.run(prompt)
    return {'response':out}

if __name__ == '__main__':
    app.run(debug=True)
