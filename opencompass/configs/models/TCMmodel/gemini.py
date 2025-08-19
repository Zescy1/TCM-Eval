
import os
from opencompass.models import OpenAISDK  
import hashlib
import time
import base64
import hmac
import datetime


gemini_url =  ''  
gemini_api = "" 

models = [
    dict(
        type=OpenAISDK, 
        path='gemini-2.5-flash', 
        key=gemini_api,
        openai_api_base=gemini_url,
        rpm_verbose=True,  
        query_per_second=0.016,
        max_seq_len=8192,
        max_out_len=2048,  
        temperature=0.01,  
        batch_size=1,
        retry=1000,
        tokenizer_path='gpt-4',
    ),
]