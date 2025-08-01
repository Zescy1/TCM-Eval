from opencompass.models import Bloom
models = [
    dict(
        type=Bloom,
        abbr='bentsao', 
        path='/home/meihan.zhang/bloom7b',
        lora_path="/home/meihan.zhang/Huatuo-Llama-Med-Chinese/lora",
        max_seq_len=8192,
        max_out_len=2048,
        run_cfg=dict(num_gpus=2),
    )
]