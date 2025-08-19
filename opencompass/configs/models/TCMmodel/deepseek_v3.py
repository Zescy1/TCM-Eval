
import os
from opencompass.models import OpenAISDK
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
        query_per_second=0.1,
        max_seq_len=8192,
        max_out_len=8192,
        temperature=0.01,
        batch_size=1,
        retry=10000,
        tokenizer_path='gpt-4',
    ),
]