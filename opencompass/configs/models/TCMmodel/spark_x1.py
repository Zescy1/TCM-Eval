# python run.py --models Spark.py --datasets medbench_gen.py --debug
import os
from opencompass.models import OpenAISDK  # 确保 OpenAISDK 已经被导入

# 科大讯飞 API 配置
spark_url = 'https://spark-api-open.xf-yun.com/v1/'  # DeepSeek API URL
spark_api = "LAivbTpXYfnrJKrBbszK:ZFSPBnXgYZfahrqdDsfL"
#spark_api = "QcjburIirhbcgFwUKMic:dzpBXOPAeoOATsVZrblK"  # 替换为你的API Key
#basin_api = "sk-a4ca50cdbe6b4f0392a0c3f36aba1314"  # 如果需要使用另一个API Key
spark_id = "dcf6ccdd"
# 获取当前时间戳和日期

models = [
    dict(
        type=OpenAISDK,  # 使用 OpenAISDK 类型
        path='lite',  # 替换为实际的模型名称
        key=spark_api,
        openai_api_base=spark_url,
        rpm_verbose=True,  
        query_per_second=0.1,
        max_seq_len=8192,
        max_out_len=2048,  # 调整为最大允许值 2048
        temperature=0.01,  # 生成温度
        batch_size=1,
        retry=10000,
        tokenizer_path='gpt-4',
       #get_headers=dynamic_get_headers,  # 动态生成请求头
    ),
]