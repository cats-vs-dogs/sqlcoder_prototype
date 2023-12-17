from langchain.document_loaders import JSONLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import sys


def create_metadata(tables, generate_file=False):

    str_desc = ""
    for table_name in tables.keys():
        str_desc += f"CREATE TABLE {table_name} (\n"
        for field_name in tables[table_name].keys():
            _type, desc = tables[table_name][field_name]
            str_desc += f"\t{field_name} {_type} --{desc}\n"
        str_desc += ")\n\n"
    if generate_file:
        f = open("slim_metadata.sql", "w")
        f.write(str_desc)
    return str_desc

def metadata_function(record: dict, metadata: dict):
    metadata["type"] = record.get("type")
    metadata["name"] = record.get("name")
    metadata["table_name"] = record.get("table_name")

    return metadata

def generate_documents(json_path, query, vecstore_name=None):
    embeddings_model = OpenAIEmbeddings()
    if vecstore_name:
        db = FAISS.load_local(vecstore_name, embeddings_model)
    else:
        loader = JSONLoader(
            file_path=json_path,
            jq_schema=".tables[].fields[]",
            content_key="description",
            metadata_func=metadata_function
        )
        data = loader.load()
        db = FAISS.from_documents(data, embeddings_model)
        db.save_local("vecstore")
    docs = db.max_marginal_relevance_search(query)
    return docs

def generate_tables(docs):
    tables = {}
    for doc in docs:
        name = doc.metadata["name"]
        _type = doc.metadata["type"]
        table_name = doc.metadata["table_name"][0]
        desc = doc.page_content
        if table_name in tables:
            tables[table_name][name] = [_type, desc]
        else:
            tables[table_name] = {}
            tables[table_name][name] = [_type, desc]
    return tables

def generate_slim(metadata_path, prompt, vecstore_name=None):
    docs = generate_documents(metadata_path, prompt, vecstore_name)
    tables = generate_tables(docs)
    metadata = create_metadata(tables)
    return metadata

