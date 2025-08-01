from mmengine.config import read_base
#opencompass examples/eval_TCMeval.py
with read_base():
    from opencompass.configs.datasets.TCMBench.tcm_choice import \
        tcmbench_datasets as tcmbench_datasets1
    from opencompass.configs.datasets.TCMBench.tcm_subjective import \
        tcmbench_datasets as tcmbench_datasets2 
    from opencompass.configs.datasets.TCMBench.tcm_depart import \
        tcmbench_datasets as tcmbench_datasets3 
    from opencompass.configs.models.TCMmodel.biancang import \
        models as biancang
    from opencompass.configs.models.TCMmodel.huatuo import \
        models as huatuo
    from opencompass.configs.models.TCMmodel.WiseDiag import \
        models as zzkj
    from opencompass.configs.models.TCMmodel.spark_x1 import \
        models as spark   
    from opencompass.configs.models.TCMmodel.baichuan3 import \
        models as baichuan
    from opencompass.configs.models.TCMmodel.qwen_max import \
        models as qw
    from opencompass.configs.models.TCMmodel.deepseek_v3 import \
        models as deepseek
    from opencompass.configs.models.TCMmodel.gpt_35 import \
        models as gpt
    from opencompass.configs.models.TCMmodel.llama3 import \
        models as llama
    from opencompass.configs.models.TCMmodel.sunsimiao import \
        models as sunsimiao
    from opencompass.configs.models.TCMmodel.qibo import \
        models as qibo
    from opencompass.configs.models.TCMmodel.mingyi import \
        models as mingyi
    from opencompass.configs.models.TCMmodel.doubao import \
        models as doubao
    from opencompass.configs.models.TCMmodel.zhipu import \
        models as zhipu 
    from opencompass.configs.models.TCMmodel.gemini import \
        models as gemini
    from opencompass.configs.models.TCMmodel.claude import \
        models as claude
    from opencompass.configs.models.TCMmodel.zhongjing import \
        models as zhongjing
    from opencompass.configs.models.TCMmodel.bentsao import \
        models as bentsao

datasets = tcmbench_datasets1+tcmbench_datasets2+tcmbench_datasets3
models=biancang+huatuo+zzkj+spark+baichuan+qw+deepseek+gpt+llama+sunsimiao+qibo+mingyi+doubao+zhipu+gemini+claude+zhongjing+bentsao
work_dir='.../opencompass/outputs/' # define which path you like
#opencompass examples/eval_TCMeval.py
#conda activate compass  
#cd opencompass

