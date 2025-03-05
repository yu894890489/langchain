import os

from dotenv import load_dotenv
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from comm.comm import Env
from comm.utils import get_db

path = os.path.join(os.path.dirname(__file__), "../env/zhiyan.env")
load_dotenv(path)

@tool
def multiply(a: int, b: int) -> int:
    """Multiply a and b."""
    return a * b

# llm = ChatOpenAI(
#     temperature=0.95,
#     model=os.getenv(Env.MODEL.value),
#     openai_api_key=os.getenv(Env.KEY.value),
#     openai_api_base=os.getenv(Env.URL.value)
# )
# from langchain import hub
# from langchain.agents import AgentExecutor, create_react_agent
#
# tools = [DuckDuckGoSearchResults()]
# prompt = hub.pull("hwchase17/react")
#
# # Choose the LLM to use
# agent = create_react_agent(llm, tools, prompt)
# agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
# invoke = agent_executor.invoke({"input": "今天的体育新闻有哪些?"})
# print(invoke)

#
# llm = ChatZhipuAI(
#     model="glm-4",
#     temperature=0.5,
# )


db = get_db()

from langchain_community.utilities import SQLDatabase

print(db.run("""
SELECT DISTINCT jbpi.id, jbpi.station_code, jbpi.cmc_antenna_num, jbpi.cmc_bbu_num, jbpi.cmc_rru_num, jbpi.cmc_transmission_num, jbpi.cmc_olt_num,
 jbpi.cmc_microwave_num, jbpi.cmc_wlan_num, jbpi.cmc_other_num, t.NAME AS cmcSysType, t1.NAME AS cmcSysPowerMode, t2.NAME AS cmcBbuIn, t3.NAME AS cmcRruUp, 
 t4.NAME AS cmcAntennaLevel, t5.NAME AS plant1High, t6.NAME AS plant2High, t7.NAME AS plant3High, t8.NAME AS plant4High, t9.NAME AS plant5High, t10.NAME AS plant6High, 
 '移动' AS NAME,jbpi.create_time FROM jy_bms_product_info jbpi LEFT JOIN jy_bms_col_tower jbct ON jbpi.station_code = jbct.station_code 
 LEFT JOIN jy_bms_col_room jbcr ON jbpi.station_code = jbcr.station_code LEFT JOIN jy_bms_col_type_sel t ON jbpi.cmc_sys_type = t.id LEFT JOIN jy_bms_col_type_sel t1
  ON jbpi.cmc_sys_power_mode = t1.id LEFT JOIN jy_bms_col_type_sel t2 ON jbpi.cmc_bbu_in = t2.id LEFT JOIN jy_bms_col_type_sel t3 ON jbpi.cmc_rru_up = t3.id 
  LEFT JOIN jy_bms_col_type_sel t4 ON jbpi.cmc_antenna_level = t4.id LEFT JOIN jy_bms_col_type_sel t5 ON jbct.plant1_high = t5.id LEFT JOIN jy_bms_col_type_sel t6 
  ON jbct.plant2_high = t6.id LEFT JOIN jy_bms_col_type_sel t7 ON jbct.plant3_high = t7.id LEFT JOIN jy_bms_col_type_sel t8 ON jbct.plant4_high = t8.id 
  LEFT JOIN jy_bms_col_type_sel t9 ON jbct.plant5_high = t9.id LEFT JOIN jy_bms_col_type_sel t10 ON jbct.plant6_high = t10.id LEFT JOIN jy_bms_col_type_sel t11 ON 
  FIND_IN_SET(t11.id, jbcr.room_use_unit) > 0 ORDER BY jbpi.create_time DESC;
"""))