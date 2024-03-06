from dotenv import load_dotenv
import os
import pandas as pd
from prompts import details, instruct 
from llama_index.core.query_engine import PandasQueryEngine

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

