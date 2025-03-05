import os

from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from comm.comm import Env, MYSQL_CONFIG
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModel,pipeline
path = os.path.join(os.path.dirname(__file__), "../env/zhiyan.env")
load_dotenv(path)
# 加载 GraphCodeBERT 模型和分词器
# model_name = "model/graphcodebert-base"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModel.from_pretrained(model_name)
model = None
@tool
def multiply(a: int, b: int) -> int:
    """Multiply a and b."""
    return a * b
from langchain.chat_models import init_chat_model

init_chat_model()
def get_llm():
    return ChatOpenAI(
        temperature=0,
        model=os.getenv(Env.MODEL.value),
        openai_api_key=os.getenv(Env.KEY.value),
        openai_api_base=os.getenv(Env.URL.value)
    )


def get_embedding():
    # 创建一个 Hugging Face Pipeline
    code_pipeline = pipeline(
        task="feature-extraction",  # 使用特征提取任务
        model=model,
        # tokenizer=tokenizer,
        device=0  # 使用 GPU（如果可用）
    )
    return HuggingFacePipeline(pipeline=code_pipeline)

def get_db():
    database_url = "mysql+pymysql://root:jinyuMysql@192.168.10.146/gansu_lanzhou_zz"
    return SQLDatabase.from_uri(database_url)


import pymysql

def get_table_schema_and_relationships():
    database = MYSQL_CONFIG.get("database")
    # 连接到 MySQL 数据库
    connection = pymysql.connect(
        host=MYSQL_CONFIG.get("host"),
        user=MYSQL_CONFIG.get("user"),
        password=MYSQL_CONFIG.get("password"),
        database=database
    )
    cursor = connection.cursor()

    # 获取表结构
    cursor.execute(f"""
        SELECT TABLE_NAME, COLUMN_NAME, DATA_TYPE, COLUMN_KEY
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = '{database}'
    """)
    columns = cursor.fetchall()

    # 获取外键关系
    cursor.execute(f"""
        SELECT 
            TABLE_NAME, COLUMN_NAME, REFERENCED_TABLE_NAME, REFERENCED_COLUMN_NAME
        FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE
        WHERE TABLE_SCHEMA = '{database}' AND REFERENCED_TABLE_NAME IS NOT NULL
    """)
    foreign_keys = cursor.fetchall()

    # 构造表结构和关系信息
    table_info = {}
    for table_name, column_name, data_type, column_key in columns:
        if table_name not in table_info:
            table_info[table_name] = {"columns": [], "primary_key": None}
        table_info[table_name]["columns"].append(f"{column_name} ({data_type})")
        if column_key == "PRI":
            table_info[table_name]["primary_key"] = column_name

    relationships = []
    for table_name, column_name, ref_table, ref_column in foreign_keys:
        relationships.append(
            f"{table_name}.{column_name} -> {ref_table}.{ref_column}"
        )

    connection.close()

    return {
        "tables": table_info,
        "relationships": relationships
    }