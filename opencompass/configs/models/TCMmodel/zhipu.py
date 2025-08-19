from opencompass.models import OpenAISDK  


zhipu_api_base = ''

zhipu_api_key = ""

models = [
    dict(
        type=OpenAISDK,
        path='glm-4-plus',
        key=zhipu_api_key, 
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