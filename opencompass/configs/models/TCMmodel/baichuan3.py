# python run.py --models wenxin.py --datasets medbench_gen.py --debug
import os
from opencompass.models import OpenAISDK# 确保 OpenAISDK 已经被导入
import tiktoken
baichuan_url = 'https://api.baichuan-ai.com/v1/'  # DeepSeek API URL,
baichuan_api = "sk-edca610945b21fb752d7d00682c21a68"  # 默认值用于调试
#sk-a4ca50cdbe6b4f0392a0c3f36aba1314

models = [
    dict(
        type=OpenAISDK,  # 使用 OpenAISDK 类型
        path='Baichuan3-Turbo',
        key=baichuan_api,
        openai_api_base=baichuan_url,
        rpm_verbose=True,  
        query_per_second=0.16,
        max_seq_len=8192,
        max_out_len=2048,
        temperature=0.01,  # 生成温度
        batch_size=1,
        retry=3,
        tokenizer_path='gpt-4',
    ),
]