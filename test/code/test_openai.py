import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from comm.comm import Env

path = os.path.join(os.path.dirname(__file__), "../../env/.env")
load_dotenv(path)
prompt = ChatPromptTemplate.from_template("tell me a short joke about {topic}")
key = os.getenv(Env.KEY.value)
os.environ['OPENAI_API_KEY'] = key
model = ChatOpenAI(model="gpt-3.5-turbo")
output_parser = StrOutputParser()

chain = prompt | model | output_parser

chain.invoke({"topic": "ice cream"})