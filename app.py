from typing import List, Tuple
from flask import Flask, request
from flask_cors import CORS
from chatbot import Chatbot
from mysql import connector
from mysql.connector import errorcode
from datetime import datetime
import mysql


def insert_entry(author: str, message: str):
    query = """
    INSERT INTO messages (conv_id, date, author, message)
    VALUES ('{}', '{}', '{}', '{}')
    """ .format(chatbot.conv_id, datetime.now(), author, message)
    cursor.execute(query)
    db_connection.commit()

def retrieve_entries(conv_id:int) -> List[Tuple[int, str]]:
    query = """
    SELECT author, message FROM messages WHERE conv_id = '{} ORDER BY date'
    """.format(conv_id)
    cursor.execute(query)
    return [x for x in cursor]

def retrieve_conversation_ids():
    query = """
    SELECT DISTINCT conv_id FROM messages
    """
    cursor.execute(query)
    return [x[0] for x in cursor]

if __name__ == "__main__":
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
    print(retrieve_conversation_ids())
    #app = Flask(__name__)
    #CORS(app)


#
#@app.route("/", methods=["GET"])
#def index():
#    #with open("conversations.pkl", "rb") as handle:
#    #    conversations = pickle.load(handle)
#    #    print(conversations)
#    #return {"conversations": [
#    #    {"id": conv["id"], "name": conv["name"]} for conv in conversations.values()
#    #]}
#
#@app.route("/loadconv", methods=["GET"])
#def get_conversation():
#    conversation = []
#    try:
#        id = request.get_json()["id"]
#        conversation = conversations[id]["conversation"]
#        memory.chat_memory.messages = conversation
#        conversation = [{"author": msg.type, "message": msg.content} for msg in conversation]
#    except:
#        # TODO
#        pass
#    return {"conversation": conversation}
#
#
#@app.route('/prompt', methods=['GET', 'POST'])
#def prompt_chatbot():
#    input = request.get_json()["prompt"]
#    memory.chat_memory.add_user_message(input)
#    try: 
#        out = agent_executor.invoke({
#             "input": input,
#             "chat_history": memory.chat_memory,
#        })["output"]
#    except:
#        out = "Sorry, I am unable to answer this question"
#    memory.chat_memory.add_ai_message(out)
#    conversation = [{"author": msg.type, "message": msg.content} for msg in memory.chat_memory.messages]
#    save_current_conversation()
#    return {"conversation": conversation}
#
#@app.route('/save', methods=["POST", "GET"])
#def save_current_conversation():
#    with open("conversations.pkl", "wb") as handle:
#        conversation = memory.chat_memory.messages
#        name = conversation[0].content if len(conversation) else "New conversation"
#        conversations[id] = {
#            "id": id,
#            "name": name,
#            "conversation": conversation
#        }
#        pickle.dump(conversations, handle, protocol=pickle.HIGHEST_PROTOCOL)
#    return {"conversations": [
#            {"id": conv["id"], "name": conv["name"]} for conv in conversations.values()
#        ],
#    }
#
#@app.route('/newconv', methods=["POST", "GET"])
#def start_conversation():
#    conversations = save_current_conversation()
#    memory.clear()
#    conversation = [{"author": msg.type, "message": msg.content} for msg in memory.chat_memory.messages]
#    return  {
#        "conversations": conversations,
#        "conversation": conversation
#    }
#