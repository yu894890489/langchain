from enum import Enum


class Env(Enum):
    URL = "OPENAI_API_BASE"
    MODEL = "OPENAI_API_MODEL"
    KEY = "OPENAI_API_KEY"


# MySQL 配置
MYSQL_CONFIG = {
    "host": "192.168.10.146",
    "user": "root",
    "password": "jinyuMysql",
    "database": "gansu_lanzhou_zz"
}

from typing_extensions import TypedDict


class State(TypedDict):
    question: str
    query: str
    result: str
    answer: str