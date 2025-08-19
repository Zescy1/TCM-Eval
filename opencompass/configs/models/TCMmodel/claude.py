# python run.py --models Spark.py --datasets medbench_gen.py --debug
import os
from opencompass.models import OpenAISDK 
import hashlib
import time
import base64
import hmac
import datetime

claude_url ='' 
claude_api = "" 

models = [
    dict(
        type=OpenAISDK,  
        path='claude-3-7-sonnet-20250219', 
        key=claude_api,
        openai_api_base=claude_url,
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