
import os
from opencompass.models import OpenAISDK
import tiktoken
z1_url = '' 
z1_api = "" 
models = [
    dict(
        type=OpenAISDK,  
        path='zzkj',
        key=z1_api,
        openai_api_base=z1_url,
        rpm_verbose=True,  
        query_per_second=0.16,
        max_seq_len=8192,
        max_out_len=2048,
        temperature=0.01, 
        batch_size=1,
        retry=100,
        tokenizer_path='gpt-4',
    ),
]