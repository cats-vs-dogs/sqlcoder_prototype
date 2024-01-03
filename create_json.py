from collections import deque
import json
import os
import sys


class Field:
    def __init__(self, name, table_name, ftype, description=None):
        self.name = name,
        self.name = self.name[0]
        self.table_name = table_name
        self.type = ftype 
        self.description = description

    def to_dict(self):
        return self.__dict__

class Table:
    def __init__(self, table_name, fields=[]):
        self.table_name = table_name,
        self.fields = fields 

    def add(self, field):
        self.fields.append(field)

    def to_dict(self):
        return {
            "table_name": self.table_name,
            "fields": [f.to_dict() for f in self.fields]
        }

    def to_json(self):
        return json.dumps({
            "table_name": self.table_name,
            "fields": [f.to_dict() for f in self.fields]
        })


def jsonify_metadata(metadata_path):
    r"""
    Convert a metadata file into a json
    and write it as "metadata.json".
    """
    with open(metadata_path) as f:
        raw_metadata = f.readlines()
    tables = deque()
    for line in raw_metadata:
        if line.startswith("CREATE TABLE "):
            chunks = line.split(" ")
            tables.appendleft(Table(chunks[2].strip()))
        if line.startswith("\t"):
            chunks = line[1:].split(",")
            field_name, field_type = chunks[0].split(" ")
            field_description = chunks[1][3:]
            tables[0].add(Field(
                field_name,
                tables[0].table_name,
                field_type.rstrip(),
                field_description.rstrip(),
            ))
    metadatajson = open("metadata.json", "w")
    metadatajson.write(json.dumps({
        'tables': [t.to_dict() for t in tables]
    }))
    metadatajson.close()

def main(argv, argc):
    if argc > 2:
        raise Exception("Provide only a single argument")
    metadata_path = argv[1]
    jsonify_metadata(metadata_path)

if __name__ == "__main__":
    main(sys.argv, len(sys.argv))

