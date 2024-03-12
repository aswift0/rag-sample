from dotenv import load_dotenv
import os
import pandas as pd
from prompts import details, instruct, context
from note_engine import note
from llama_index.core.query_engine import PandasQueryEngine
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from llama_index.llms import openai

load_dotenv()

#no pyspark
#pd to load data
car_data = os.path.join("reference", "2023 Car Dataset.csv")
df = pd.read_csv(car_data)

#query engine for pandas df
interface = PandasQueryEngine(df=df, verbose=True, instruction_str=instruct)
#context for prompt attached
interface.update_prompts({"pandas_prompt":details})

interface.query("How many Subaru's were bought in 2023?")

tools = [
    note,
    QueryEngineTool(query_engine=interface, metadata = ToolMetadata(
    name = "car_info",
    description = "data about cars metrics from 2023")
    )
]

model = openai(model = "gpt-3.5-turbo-0613")
agent = ReActAgent.from_tools(tools, llm=model, verbose=True, context = context)

while(prompt := input("Ask me questions about cars! X to quit")) != "X":
    res = agent.query(prompt)
    print(res)

