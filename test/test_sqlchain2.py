from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers.openai_tools import PydanticToolsParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from comm.utils import get_llm, get_db

llm = get_llm()
db = get_db()

from langchain_community.utilities import SQLDatabase

table_info = db.get_table_info(table_names=["jy_bms_station"])
# print(db.run("SELECT * FROM jy_bms_station LIMIT 10;"))
print(db.get_usable_table_names())
class Table(BaseModel):
    """Table in SQL database."""

    name: str = Field(description="Name of table in SQL database.")


table_info = "\n".join(table_info)
system = f"""Return the names of ALL the SQL tables that MIGHT be relevant to the user question. \
The tables are:

{table_info}

Remember to include ALL POTENTIALLY RELEVANT tables, even if you're not sure that they're needed."""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{input}"),
    ]
)
# prompt = hub.pull("rlm/text-to-sql")
llm_with_tools = llm.bind_tools([Table])
# output_parser = PydanticToolsParser(tools=[Table])
output_parser = StrOutputParser(tools=[Table])

table_chain = prompt | llm_with_tools | output_parser

out = table_chain.invoke({"input": "请分析下甘肃省所有城市站址总量的排名"})
print(out)
