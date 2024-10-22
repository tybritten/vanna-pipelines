from vanna.openai import OpenAI_Chat
from vanna.chromadb import ChromaDB_VectorStore
from openai import OpenAI
from loguru import logger
from vanna.flask import VannaFlaskApp
import os

CHAT_MODEL = os.environ.get("CHAT_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v0.4")
CHAT_MODEL_BASE_URL = os.environ.get("CHAT_MODEL_BASE_URL", None)
TEMP = os.environ.get("TEMP", 0.4)
MAX_TOKENS = os.environ.get("MAX_TOKENS", 131072)
DB_PATH = os.environ.get("DB_PATH", "./db")
DATABASE_CONNECTION_STRING = os.environ.get("DATABASE_CONNECTION_STRING")


class MyVanna(ChromaDB_VectorStore, OpenAI_Chat):
    def __init__(self, config={}, client=None):
        ChromaDB_VectorStore.__init__(self, config=config)
        OpenAI_Chat.__init__(self, client=client, config=config)


client = OpenAI(base_url=CHAT_MODEL_BASE_URL, api_key="fake")
logger.info(f"Using model {CHAT_MODEL} with temp {TEMP} and max_tokens {MAX_TOKENS} at url {CHAT_MODEL_BASE_URL}")
vn = MyVanna(client=client, config={'model': CHAT_MODEL, 'path': DB_PATH, 'max_tokens': MAX_TOKENS})

vn.connect_to_mssql(odbc_conn_str=DATABASE_CONNECTION_STRING)
training_data = vn.get_training_data()
if training_data.count().id == 0:
    logger.info("empty vector db, initializing..")
    df_information_schema = vn.run_sql("SELECT * FROM INFORMATION_SCHEMA.COLUMNS")
    plan = vn.get_training_plan_generic(df_information_schema)
    vn.train(plan=plan)
else:
    logger.info("vector db already initialized, skipping..")
    logger.info(f"vector db has {training_data.count().id} records")

app = VannaFlaskApp(vn, allow_llm_to_see_data=True)

app.run()
