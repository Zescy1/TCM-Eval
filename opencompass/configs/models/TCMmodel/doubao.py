import os
from opencompass.models import OpenAISDK# 确保 OpenAISDK 已经被导入
import tiktoken
gpt_url = 'https://ark.cn-beijing.volces.com/api/v3'  # DeepSeek API URL,
gpt_api = "ac91ac18-7067-41cd-b6aa-7203c72eddbe"  # 默认值用于调试
#sk-a4ca50cdbe6b4f0392a0c3f36aba1314

models = [
    dict(
        type=OpenAISDK,  # 使用 OpenAISDK 类型
        path='doubao-seed-1-6-250615',  # 模型路径
        key=gpt_api,
        openai_api_base=gpt_url,
        rpm_verbose=True,  
        query_per_second=0.1,
        max_seq_len=8192,
        max_out_len=8192,
        temperature=0.01,  # 生成温度
        batch_size=1,
        retry=10000,
        tokenizer_path='gpt-4',
    ),
]