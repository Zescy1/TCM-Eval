
import os
from opencompass.models import OpenAISDK  


qwen_url = ''
qwen_api_key = "" 

# 定义模型配置
models = [
    dict(
        type=OpenAISDK,  
        path='qwen-plus',  
        key=qwen_api_key,
        openai_api_base=qwen_url,
        query_per_second=0.16,  
        max_seq_len=8192,  
        max_out_len=8192,  
        temperature=0.01, 
        batch_size=1,
        retry=3,
        tokenizer_path='gpt-4',
    ),
]
