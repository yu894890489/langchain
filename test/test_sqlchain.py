from typing import TypedDict, Annotated

from langchain_community.utilities import SQLDatabase
from langchain.prompts import PromptTemplate
from sqlalchemy import create_engine

from comm.comm import State
from comm.utils import get_llm, get_table_schema_and_relationships
from langchain_experimental.sql import SQLDatabaseChain

# 初始化数据库连接
database_url = "mysql+pymysql://root:jinyuMysql@192.168.10.146/gansu_lanzhou_zz"
# engine = create_engine(database_url)
# db = SQLDatabase(engine)
db = SQLDatabase.from_uri(database_url)
# # 自定义提示模板
# sql_prompt_template = PromptTemplate(
#     input_variables=["question", "table_info"],
#     template="根据以下问题和表结构及关系信息生成 SQL 查询：\n"
#              "表结构及关系：\n{table_info}\n"
#              "问题：{question}\n"
#              "SQL 查询："
# )

# 初始化 LLM 和 SQLDatabaseChain
llm = get_llm()

from langchain import hub

query_prompt_template = hub.pull("langchain-ai/sql-query-system-prompt")

class QueryOutput(TypedDict):
    """Generated SQL query."""

    query: Annotated[str, ..., "Syntactically valid SQL query."]


def write_query(state: State):
    """Generate SQL query to fetch information."""
    prompt = query_prompt_template.invoke(
        {
            "dialect": db.dialect,
            "top_k": 10,
            "table_info": db.get_table_info(),
            "input": state["question"],
        }
    )
    structured_llm = llm.with_structured_output(QueryOutput)
    result = structured_llm.invoke(prompt)
    return {"query": result["query"]}

print(write_query({"question": "请分析下甘肃省所有城市站址总量的排名"}))

#
# db_chain = SQLDatabaseChain.from_llm(db=db, verbose=True,llm=llm,prompt=sql_prompt_template)
#
# # 示例调用
# question = "请分析下甘肃省所有城市站址总量的排名。"
# table_info = get_table_schema_and_relationships()
# # db_chain.run(question=question, table_info=table_info,llm=llm,prompt=sql_prompt_template)
# sql_query = db_chain.invoke(question=question, table_info=table_info)
# print(f"生成的 SQL 查询：{sql_query}")