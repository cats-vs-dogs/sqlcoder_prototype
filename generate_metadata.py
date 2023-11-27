import sys
import sqlite3
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.chains import LLMChain



def get_schemas(db_path):
    schemas = {}
    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    table_names = cursor.fetchall()
    for raw_table_name in table_names:
        table_name = raw_table_name[0]
        schemas[table_name] = get_schema(cursor, table_name)
    return schemas


def get_schema(cursor, table_name):
    table_schema = {}
    cursor.execute(f"PRAGMA table_info('{table_name}')")
    schema = cursor.fetchall()
    for col in schema:
        table_schema[col[1]] = col[2]
    return table_schema


def generate_metadata_file(schema):
    file = open("metadata.sql", "w")
    prompt = PromptTemplate.from_template(
        """
        Generate a brief one sentence description
        of what you think an SQL column with name 
        {col_name} that holds a value of type 
        {col_type} in a table with name {table_name}
        means. Note that this is a from a loan portfolio
        database.
        """
    )
    description_chain = LLMChain(llm=OpenAI(), prompt=prompt, output_parser=StrOutputParser())
    for table_name in schema.keys():
        generate_metadata(schema, table_name, file, description_chain)
    file.close()
        
        
def generate_metadata(schema, table_name, file, description_chain=None):
    file.write(f"CREATE TABLE {table_name} (\n")
    for k, v in schema[table_name].items():
        if description_chain:
            out = description_chain.invoke({
                "col_name"   : k,
                "col_type"   : v,
                "table_name" : table_name,
            })
            desc = out["text"]
        else:
            desc = "no description is given"
        file.write(f"\t{k} {v}, --{desc.strip()} \n")
    file.write(f");\n\n")


def main(argv, arc):
    if arc>2:
        raise Exception("Provide only a single argument")
    elif arc==1:
        raise Exception("Provide a file path")
    db_path = argv[1]
    schemas = get_schemas(db_path)
    generate_metadata_file(schemas)



if __name__ == "__main__":
    main(sys.argv, len(sys.argv))


