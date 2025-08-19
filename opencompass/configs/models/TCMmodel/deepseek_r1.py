
import os
from opencompass.models import OpenAISDK# 确保 OpenAISDK 已经被导入
import tiktoken
deepseek_url = ''  
deepseek_api = ""  
models = [
    dict(
        type=OpenAISDK,  
        path='deepseek-chat',
        key=deepseek_api,
        openai_api_base=deepseek_url,
        rpm_verbose=True,  
        query_per_second=0.16,
        max_seq_len=8192,
        max_out_len=8192,
        temperature=0.01, 
        batch_size=1,
        retry=10,
        tokenizer_path='gpt-4',
    ),
]