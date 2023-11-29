import sys
import sqlite3
import pandas as pd


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


def generate_metadata_file(schema, descriptions):
    file = open("metadata.sql", "w")
    for table_name in schema.keys():
        generate_metadata(schema, table_name, file, descriptions)
    file.close()

def parse_descriptions(file_path):
    df = pd.read_csv(file_path, sep=";", encoding="iso_8859_1")
    k = df["COLUMN_NAME"].str.upper().to_list()
    v = df["COLUMN_DESCRIPTION_EN"].to_list()
    return dict(zip(k, v))
        
        
def generate_metadata(schema, table_name, file, descriptions):
    file.write(f"CREATE TABLE {table_name} (\n")
    for k, v in schema[table_name].items():
        desc = "NO DESCRIPTION WAS FOUND"
        if k in descriptions:
            desc = str(descriptions[k])
            desc = desc.replace("\n", "")
        file.write(f"\t{k} {v}, --{desc.rstrip()} \n")
    file.write(f");\n\n")


def main(argv, arc):
    if arc>3:
        raise Exception("Provide only a single argument")
    elif arc==1:
        raise Exception("Provide a file path")
    db_path = argv[1]
    desc_path = argv[2]
    schemas = get_schemas(db_path)
    descriptions = parse_descriptions(desc_path)
    generate_metadata_file(schemas, descriptions)



if __name__ == "__main__":
    main(sys.argv, len(sys.argv))


