from opencompass.models import Ollama

models = [
    dict(
        type=Ollama,
        abbr='llama',                # 模型简称（用于输出日志和结果）
        path='ollama',                # 固定值，表示使用 ollama 客户端
        model='llama3.3:latest',               # 指定模型名称，必须与 ollama list 中一致
        api_base='http://localhost:11434/api/generate',  # Ollama API 地址
        max_seq_len=9182,             # 最大输入长度
        max_out_len=9182,             # 最大生成长度
        batch_size=1,                 # 批处理大小（Ollama 默认不支持批量推理）
        generation_kwargs=dict(
            temperature=0.01,          # 控制生成多样性              # nucleus sampling
        ),
        query_per_second=5,           # QPS 限速（可选）
    )
]