import os
import streamlit as st
from vanna.openai import OpenAI_Chat
from vanna.chromadb import ChromaDB_VectorStore
from openai import OpenAI


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



@st.cache_resource(ttl=3600)
def setup_vanna():
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
    return vn

@st.cache_data(show_spinner="Generating sample questions ...")
def generate_questions_cached():
    vn = setup_vanna()
    return vn.generate_questions()


@st.cache_data(show_spinner="Generating SQL query ...")
def generate_sql_cached(question: str):
    vn = setup_vanna()
    return vn.generate_sql(question=question, allow_llm_to_see_data=True)

@st.cache_data(show_spinner="Checking for valid SQL ...")
def is_sql_valid_cached(sql: str):
    vn = setup_vanna()
    return vn.is_sql_valid(sql=sql)

@st.cache_data(show_spinner="Running SQL query ...")
def run_sql_cached(sql: str):
    vn = setup_vanna()
    return vn.run_sql(sql=sql)

@st.cache_data(show_spinner="Checking if we should generate a chart ...")
def should_generate_chart_cached(question, sql, df):
    vn = setup_vanna()
    return vn.should_generate_chart(df=df)

@st.cache_data(show_spinner="Generating Plotly code ...")
def generate_plotly_code_cached(question, sql, df):
    vn = setup_vanna()
    code = vn.generate_plotly_code(question=question, sql=sql, df=df)
    return code


@st.cache_data(show_spinner="Running Plotly code ...")
def generate_plot_cached(code, df):
    vn = setup_vanna()
    return vn.get_plotly_figure(plotly_code=code, df=df)


@st.cache_data(show_spinner="Generating followup questions ...")
def generate_followup_cached(question, sql, df):
    vn = setup_vanna()
    return vn.generate_followup_questions(question=question, sql=sql, df=df)

@st.cache_data(show_spinner="Generating summary ...")
def generate_summary_cached(question, df):
    vn = setup_vanna()
    return vn.generate_summary(question=question, df=df)
