from opencompass.models import HuaweiModel

models = [
    dict(
        type=HuaweiModel,          # 使用 HuaweiModel 类型         
        abbr='qibo', 
        path='Qibo',           # 模型简称（用于输出日志和结果）
        api_base='http://localhost:5000/tohuawei',
        max_seq_len=9182,             # 最大输入长度
        max_out_len=9182,             # 最大生成长度
        batch_size=1,                 # 批处理大小（Ollama 默认不支持批量推理）
        generation_kwargs=dict(
            temperature=0.01,          # 控制生成多样性              # nucleus sampling
        ),
        query_per_second=0.1,           # QPS 限速（可选）
    )
]