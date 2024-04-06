from typing import List, Tuple
from flask import Flask, request
from flask_cors import CORS
from chatbot import Chatbot
from mysql import connector
from mysql.connector import errorcode
from datetime import datetime
import mysql


def insert_entry(author: str, message: str):
    message.replace("'", "`")
    sql = """
    INSERT INTO messages (conv_id, date, author, message)
    VALUES (%s, %s, %s, %s)
    """
    vals = (chatbot.conv_id, datetime.now(), author, message)
    cursor.execute(sql, vals)
    db_connection.commit()

def retrieve_entries(conv_id:int) -> List[Tuple[int, str]]:
    query = """
    SELECT author, message FROM messages WHERE conv_id = '{}' ORDER BY date
    """.format(conv_id)
    cursor.execute(query)
    return [x for x in cursor]

def retrieve_conversation_id_names() -> List[Tuple[int, str]]:
    query = """
    SELECT conv_id, message 
    FROM( SELECT messages.*, MIN(id) OVER (PARTITION BY conv_id) AS min
        FROM messages) AS tt
    WHERE id = min 
    """
    cursor.execute(query)
    return [x for x in cursor]

config = {
    "host":"localhost",
    "user":"chatbot",
    "password":"",
    "database":"chatbot"
}
try:
    db_connection = connector.connect(**config)
    cursor = db_connection.cursor()
except mysql.connector.Error as err:
    if err.errno == errorcode.ER_BAD_DB_ERROR:
        del config["database"]
        db_connection = connector.connect(**config)
        cursor = db_connection.cursor()
        cursor.execute("CRETE DATABASE chatbot")
    else:
        print("Database connection failed with the following error:\n{}".format(err))
cursor.execute("SHOW TABLES")
tables = [x[0] for x in cursor]
if "messages" not in tables:
    cursor.execute(
    """
    CREATE TABLE messages (
        id INT unsigned NOT NULL AUTO_INCREMENT PRIMARY KEY,
        conv_id INT unsigned NOT NULL,
        date DATETIME NOT NULL,
        author VARCHAR(100) NOT NULL,
        message VARCHAR(10000) NOT NULL
    )
    """)
chatbot = Chatbot()
app = Flask(__name__)
CORS(app)

@app.route("/", methods=["GET"])
def index():
    entries = retrieve_conversation_id_names()
    response = [
        {"conv_id": e[0], "message": e[1]} for e in entries 
    ]
    print(response)
    return {
        "repsonse": response 
    } 
    
@app.route("/inference", methods=["GET"])
def inference():
    input = request.get_json()["prompt"]
    insert_entry(author="User", message=input)
    out = chatbot.run(input)
    insert_entry(author="AI", message=out)
    return {
        "response": retrieve_entries(chatbot.conv_id)
    }

@app.route("/new", methods=["GET", "POST"])
def new_conv():
    chatbot.start_conversation()
    return {
        "response": retrieve_entries(chatbot.conv_id)
    }
