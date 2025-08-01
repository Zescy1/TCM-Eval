from opencompass.models import OpenAISDK  # 导入改进后的适配器

# 智谱AI的API基础URL
zhipu_api_base = 'https://open.bigmodel.cn/api/paas/v4/'
# 你的智谱API密钥
zhipu_api_key = "c09f97c34d8841d484e6fd7aec28e4c9.BnHyWukPT34tpZnv"

models = [
    dict(
        type=OpenAISDK,
        path='glm-4-plus',
        key=zhipu_api_key,  # 确保key参数正确传递
        openai_api_base=zhipu_api_base,
        rpm_verbose=True,
        query_per_second=0.1,
        max_seq_len=8192,
        max_out_len=7967,
        temperature=0.01,
        batch_size=1,
        retry=10,
        tokenizer_path='gpt-4',
    ),
]