# python run.py --models Spark.py --datasets medbench_gen.py --debug
import os
from opencompass.models import OpenAISDK  # 确保 OpenAISDK 已经被导入
import hashlib
import time
import base64
import hmac
import datetime

# 科大讯飞 API 配置
gemini_url =  'https://api.key77qiqi.cn/v1'  # DeepSeek API URL
gemini_api = "sk-qfgHZjQQt0qYrO0p49BaOuTp2t1eyDxrhQpr9blZzCZibevS" 

models = [
    dict(
        type=OpenAISDK,  # 使用 OpenAISDK 类型
        path='gemini-2.5-flash',  # 替换为实际的模型名称
        key=gemini_api,
        openai_api_base=gemini_url,
        rpm_verbose=True,  
        query_per_second=0.016,
        max_seq_len=8192,
        max_out_len=2048,  # 调整为最大允许值 2048
        temperature=0.01,  # 生成温度
        batch_size=1,
        retry=1000,
        tokenizer_path='gpt-4',
       #get_headers=dynamic_get_headers,  # 动态生成请求头
    ),
]