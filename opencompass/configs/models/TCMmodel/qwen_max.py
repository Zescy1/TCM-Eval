
# python run.py --models deepseek_r1.py --datasets demo_cmmlu_chat_gen.py --debug
# python run.py --models qwq_32b.py --datasets medbench_gen.py --debug
import os
from opencompass.models import OpenAISDK  # 使用 Qwen 类

# 阿里云 DashScope 配置
qwen_url = 'https://api.key77qiqi.cn/v1'
qwen_api_key = "sk-qfgHZjQQt0qYrO0p49BaOuTp2t1eyDxrhQpr9blZzCZibevS" # 替换为您的真实 API Key

# 定义模型配置
models = [
    dict(
        type=OpenAISDK,  # 使用 Qwen 类型
        path='qwen-plus',  # 模型名称，例如 qwen-max 或 qwen-turbo
        key=qwen_api_key,
        openai_api_base=qwen_url,
        query_per_second=0.16,  # 每秒允许的最大查询次数
        max_seq_len=8192,  # 最大输入序列长度
        max_out_len=8192,  # 最大输出序列长度
        temperature=0.01,  # 生成温度
        batch_size=1,
        retry=3,
        tokenizer_path='gpt-4',
    ),
]
