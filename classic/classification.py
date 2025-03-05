from langchain.chains import LLMChain, RouterChain
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
import pandas as pd
import matplotlib.pyplot as plt
from langchain_community.tools import DuckDuckGoSearchRun
from sqlalchemy import create_engine

from comm.utils import get_llm, get_db

# 初始化大模型
llm = ChatOpenAI(temperature=0, model_name="gpt-4")

# ===== 1. 问题分类器 =====
classification_template = """将用户问题分类到以下三类之一：
1. business_query - 需要访问数据库的业务问题，涉及数据查询分析
2. realtime_query - 需要实时网络信息的时效性问题
3. reasoning_query - 需要逻辑推理的开放性问题

只返回分类名称，不要任何解释。示例：
问题：去年各季度销售额对比如何？
分类：business_query

当前问题：{input}
分类结果："""

classification_prompt = PromptTemplate(
    template=classification_template,
    input_variables=["input"],
    output_parser=RouterOutputParser()
)
llm = get_llm()
str_output_parser = StrOutputParser()
classifier_chain =  classification_prompt|llm|str_output_parser


# ===== 2. 各处理模块 =====
class BusinessHandler:
    def __init__(self):
        self.db = get_db()
        database_url = "mysql+pymysql://root:jinyuMysql@192.168.10.146/gansu_lanzhou_zz"
        self.engine = create_engine(database_url)

    def run(self, query):
        # 自然语言转SQL
        sql_prompt = f"""基于数据库结构：
        {self.db.get_table_info()}
        生成查询SQL：{query}"""

        sql = llm.invoke(sql_prompt)
        print(sql)
        # 执行查询
        df = pd.read_sql(sql, self.engine)

        # 自动生成图表
        if len(df) > 0:
            df.plot(kind='bar')
            plt.title('查询结果可视化')
            plt.savefig('temp_chart.png')

        # 数据分析报告
        analysis_prompt = f"""数据：
        {df.to_markdown()}

        请分析数据趋势，并总结关键发现："""
        return llm.predict(analysis_prompt)


class RealtimeHandler:
    def __init__(self):
        self.search = DuckDuckGoSearchRun()

    def run(self, query):
        results = self.search.run(query)
        summary_prompt = f"""基于以下实时搜索结果：
        {results[:2000]}  # 限制输入长度

        请生成简洁的中文摘要，包含关键信息点："""
        return llm.predict(summary_prompt)


class ReasoningHandler:
    def run(self, query):
        reasoning_prompt = f"""你是一个专业分析师，请仔细思考后回答：
        问题：{query}
        逐步分析："""
        return llm.predict(reasoning_prompt)


# ===== 3. 路由控制 =====
handler_mapping = {
    "business_query": BusinessHandler().run,
    "realtime_query": RealtimeHandler().run,
    "reasoning_query": ReasoningHandler().run
}


def process_query(query):
    # Step 1: 分类问题
    category = classifier_chain.invoke({"input":query}).strip().lower()

    # Step 2: 路由到对应处理器
    handler = handler_mapping.get(category, handler_mapping["reasoning_query"])
    return handler(query)


# 使用示例
queries = [
    "请分析下甘肃省所有城市站址总量的排名",  # 业务问题
    "今天硅谷发生了什么重要科技新闻",  # 实时问题
    "如何提高团队协作效率的理论框架"  # 推理问题
]

for q in queries:
    print(f"问题：{q}")
    print(f"回答：{process_query(q)}\n{'-' * 40}")