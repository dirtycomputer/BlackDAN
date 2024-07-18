import os

from openai import OpenAI


OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
ALI_API_KEY = os.getenv('ALI_API_KEY')
LLAMA_API_KEY = os.getenv('LLAMA_API_KEY')
YI_API_KEY = os.getenv('YI_API_KEY')

yi_client = OpenAI(
    api_key=YI_API_KEY, 
    base_url="https://api.lingyiwanwu.com/v1",
)

ali_client = OpenAI(
    api_key=ALI_API_KEY, 
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

llama_client = OpenAI(
    api_key=LLAMA_API_KEY, 
    base_url="https://api.llama-api.com",
)

local_client = OpenAI(
    base_url="https://localhost:8000/v1",
)

openai_client = OpenAI(
    api_key=OPENAI_API_KEY
)