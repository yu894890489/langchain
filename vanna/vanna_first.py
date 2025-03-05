import os

from dotenv import load_dotenv
from vanna import weaviate
from vanna.ZhipuAI import ZhipuAI_Chat
from vanna.weaviate import WeaviateDatabase
from weaviate.auth import AuthApiKey

from comm.comm import Env, MYSQL_CONFIG


def get_vector_store():
    client = weaviate.connect_to_custom(
        skip_init_checks=False,
        http_host="192.168.229.128",
        http_port=8080,
        http_secure=False,
        grpc_host="192.168.229.128",
        grpc_port=50051,
        grpc_secure=False,
        # 对应 AUTHENTICATION_APIKEY_ALLOWED_KEYS 中的密钥
        # 注意：此处只需要密钥即可，不需要用户名称
        auth_credentials=AuthApiKey("test-secret-key")
    )

    # STORE = WeaviateVectorStore(
    #     weaviate_client=client, index_name="LlamaIndex"
    # )
    # return STORE
path = os.path.join(os.path.dirname(__file__), "../env/zhiyan.env")
load_dotenv(path)
class MyVanna(WeaviateDatabase, ZhipuAI_Chat):
    def __init__(self, vectorConfig=None,config=None):
        WeaviateDatabase.__init__(self, config=vectorConfig)
        ZhipuAI_Chat.__init__(self, config=config)

vectorConfig = {"weaviate_api_key":AuthApiKey("test-secret-key"),
                "weaviate_url":"weaviate_api_key",
                "weaviate_port":8080,"weaviate_grpc":50051}
llm_config = {'api_key': os.getenv(Env.KEY.value), 'model': os.getenv(Env.MODEL.value)}
vn = MyVanna(vectorConfig,llm_config)
vn.connect_to_mysql(host=MYSQL_CONFIG.host, dbname=MYSQL_CONFIG.database, user=MYSQL_CONFIG.user, password=MYSQL_CONFIG.password, port=3306)
# The information schema query may need some tweaking depending on your database. This is a good starting point.
# 训练dll数据，只需要训练一次
# df_information_schema = vn.run_sql("SELECT * FROM INFORMATION_SCHEMA.COLUMNS")
with open('../sql/data/345创建表的schema.txt', 'r',encoding="utf-8") as file:
    content = file.read()
    vn.train(ddl=content)

# This will break up the information schema into bite-sized chunks that can be referenced by the LLM
plan = vn.get_training_plan_generic(df_information_schema)
vn.train(plan)
