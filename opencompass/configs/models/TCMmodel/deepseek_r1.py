
# python run.py --models deepseek_r1.py --datasets demo_cmmlu_chat_gen.py --debug
import os
from opencompass.models import OpenAISDK# 确保 OpenAISDK 已经被导入
import tiktoken
deepseek_url = 'https://api.deepseek.com/v1'  # DeepSeek API URL,
deepseek_api = "sk-a0b765c9a0734d50bb61d8038c614eab"  # 默认值用于调试
#sk-a4ca50cdbe6b4f0392a0c3f36aba1314

models = [
    dict(
        type=OpenAISDK,  # 使用 OpenAISDK 类型
        path='deepseek-chat',
        key=deepseek_api,
        openai_api_base=deepseek_url,
        rpm_verbose=True,  
        query_per_second=0.16,
        max_seq_len=8192,
        max_out_len=8192,
        temperature=0.01,  # 生成温度
        batch_size=1,
        retry=10,
        tokenizer_path='gpt-4',
    ),
]