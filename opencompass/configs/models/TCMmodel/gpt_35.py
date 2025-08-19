

import os
from opencompass.models import OpenAISDK
import tiktoken
gpt_url = ''  
gpt_api = "" 


models = [
    dict(
        type=OpenAISDK,  
        path='gpt-4',  
        key=gpt_api,
        openai_api_base=gpt_url,
        rpm_verbose=True,  
        query_per_second=0.1,
        max_seq_len=8192,
        max_out_len=8192,
        temperature=0.01, 
        batch_size=1,
        retry=100,
        tokenizer_path='gpt-4',
    ),
]