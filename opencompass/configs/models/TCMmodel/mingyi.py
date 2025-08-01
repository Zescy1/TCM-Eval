from opencompass.models import QwenLM
models = [
    dict(
        type=QwenLM,
        abbr='mingyi',
        path='/home/meihan.zhang/MINGYI',
        max_seq_len=8192,
        max_out_len=2048,
        run_cfg=dict(num_gpus=1),
    )
]
