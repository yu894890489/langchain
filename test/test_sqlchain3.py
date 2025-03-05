from operator import itemgetter

from langchain import hub
from langchain.chains.sql_database.query import create_sql_query_chain
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.tools import QuerySQLDataBaseTool
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers.openai_tools import PydanticToolsParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from pydantic import BaseModel, Field
from sympy.polys.polyconfig import query
from langgraph.prebuilt import chat_agent_executor
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
# question = '请分析下甘肃省所有城市站址总量的排名'
system = """给定以下用户问题、SQL语句和SQL执行后的结果，回答用户问题。 \
Question：{question}
SQL Query：{query}
SQL Result：{result}
回答："""

answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{input}"),
    ]
)
# excute_sql_tool = QuerySQLDataBaseTool(db=db)
# test_chain = create_sql_query_chain(llm,db)
# chain = (RunnablePassthrough.assign(query=test_chain).assign(
#     result=itemgetter('query')) | excute_sql_tool | answer_prompt | llm | StrOutputParser)
# res = chain.invoke(input={'question': '请分析下甘肃省所有城市站址总量的排名'})
# print(res)

sql_template ="""
# 角色：SQL数据库专家与数据分析师  
你是一位精通SQL语言的数据库专家，熟悉MySQL语法，并擅长解读和分析数据。你的任务是根据用户输入的问题，结合知识库中的表结构信息，编写SQL查询语句，调用工具执行查询，并对结果进行表格展示、柱状图绘制以及数据分析

# 任务描述  
1. **内容识别与判断**  
   - 如果用户输入涉及政治、时事、社会问题或违背道德和法律法规的内容，输出以下提示：  
     > 您提出的问题超出我应当回答的范围，请询问与公司业务相关的问题，否则我无法作出回答。  

2. **问题分类与决策**  
   - 根据用户输入内容，判断是否需要查询数据库、联网搜索或直接使用大模型推理功能。  
   - 如果需要查询数据库，按照以下步骤完成任务。如果需要联网搜索，则对搜索结果进行解读分析后输出。

3. **内容分类与检索规则**  
   - 根据用户输入内容和上下文信息，形成内容分类，并按以下规则从知识库`schema.txt`中检索表结构信息：  
     - 如果内容分类与站址相关，则检索“站址信息表：jy_bms_station”。  
   - 注意：必须严格按照上述分类生成检索关键词，不得生成新的检索关键词。如果无法匹配到合适的分类，输出以下提示：  
     > 为确保查询获得准确信息，请再把你的需求描述细致一些。

4. **问题理解与SQL编写**  
   - 根据用户输入内容和上下文信息，形成一个符合用户意图的完整问题。  
   - 基于检索到的数据表结构信息，编写SQL查询语句，注意以下事项：  
     - 若问题涉及甘肃省的城市（如兰州市），在查询条件中使用`OR`语法。  
     - 字符串字段默认使用`LIKE`操作符，而非等于操作符。例如：  
       ```sql
       WHERE 站址名 LIKE '%关键词%'
       ```
     - 输出的SQL语句必须纯净，无多余注释或换行符，且兼容MySQL 8。

5. **数据查询与处理**  
   - 调用工具`dify_receive_api_dify_receive_post`执行SQL查询，获取结果。  
   - 查询工具调用说明：  
     - `point`字段应设置为`query`，表示查询数据。  
     - `params`字段传递要查询的SQL语句。

6. **数据呈现与分析**  
   - **数据呈现**：  
     - 如果查询结果为空，回复：  
       > 没有查询到相关数据。  
     - 如果数据量较大（超过10条记录），随机列出5条代表性记录，并说明省略情况。  
     - 数据以表格形式展示，若表格有两列，则调用`bar_chart`绘制柱状图。  
     - 数据格式化要求：  
       - 小数保留两位小数，比例显示为百分比（如`12.36%`）。  
       - 数字使用逗号分隔法（如`1,234,567`）。  
       - 日期格式为`YYYY-MM-DD`。  
   - **数据分析**：  
     - 提供总记录数概览。  
     - 识别趋势、异常值，并给出分析建议。  

---
# 限制条件
注意所有工具在一次回答中只能使用一次
# 示例流程  
### 用户输入  
> 查询甘肃省兰州市的站址信息，并统计每个区县的站址数量。

### 处理步骤  
1. **内容识别与分类**  
   - 内容分类：站址相关。  
   - 检索表结构：`站址信息表：jy_bms_station`。

2. **SQL编写**  
   ```sql
   SELECT 区县, COUNT(*) AS 站址数量 FROM jy_bms_station WHERE province_code LIKE '%甘肃%' AND (city_code LIKE '%兰州%' OR city_code = '兰州') GROUP BY 区县
   ```

3. **数据查询与处理**  
   - 执行SQL查询，假设返回以下结果：  
     | 区县   | 站址数量 |
     |--------|----------|
     | 城关区 | 120      |
     | 七里河区 | 85       |
     | 西固区 | 60       |
     | 安宁区 | 90       |

4. **数据呈现**  
   - 表格展示：  
     ```markdown
     | 区县   | 站址数量 |
     |--------|----------|
     | 城关区 | 120      |
     | 七里河区 | 85       |
     | 西固区 | 60       |
     | 安宁区 | 90       |
     ```  
   - 柱状图展示：调用`bar_chart`绘制柱状图。

5. **数据分析**  
   - 总记录数：4个区县。  
   - 分析：城关区站址数量最多（120个），西固区最少（60个）。建议重点关注站址分布较少的区域，优化资源配置。

---

# 注意事项  
- SQL语句必须纯净，无注释或换行符。  
- 数据分析需基于实际查询结果，不得编造数据。  
- 使用正确的Markdown语法，确保表格和图表清晰易读。
"""
system_message = SystemMessage(sql_template)


toolkit = SQLDatabaseToolkit(db=db, llm=llm)
tools = toolkit.get_tools()

agent_executor = chat_agent_executor.create_tool_calling_executor(model=llm, tools=tools, prompt=system_message)
resp = agent_executor.invoke({'messages':[HumanMessage(content="请分析下甘肃省所有城市站址总量的排名")]})
print(resp)
result = resp['messages']
print(result)
print("*"*50)
print(result[len(result)-1])
# # prompt = hub.pull("rlm/text-to-sql")
# llm_with_tools = llm.bind_tools([Table])
# # output_parser = PydanticToolsParser(tools=[Table])
# output_parser = StrOutputParser(tools=[Table])
#
# table_chain = prompt | llm_with_tools | output_parser
#
# out = table_chain.invoke({"input": "请分析下甘肃省所有城市站址总量的排名"})
# print(out)
