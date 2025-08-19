from opencompass.models import Bloom
models = [
    dict(
        type=Bloom,
        abbr='bentsao', 
        path='/home/bloom7b',
        lora_path="/home/Huatuo-Llama-Med-Chinese/lora",
        max_seq_len=8192,
        max_out_len=2048,
        run_cfg=dict(num_gpus=2),
    )
]