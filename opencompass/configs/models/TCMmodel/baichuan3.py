
import os
from opencompass.models import OpenAISDK
import tiktoken
baichuan_url = '' 
baichuan_api = "" 


models = [
    dict(
        type=OpenAISDK,  
        path='Baichuan3-Turbo',
        key=baichuan_api,
        openai_api_base=baichuan_url,
        rpm_verbose=True,  
        query_per_second=0.16,
        max_seq_len=8192,
        max_out_len=2048,
        temperature=0.01, 
        batch_size=1,
        retry=3,
        tokenizer_path='gpt-4',
    ),
]