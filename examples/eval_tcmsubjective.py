from mmengine.config import read_base
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
with read_base():
#from opencompass.configs.datasets.TCMBench.tcmbench_gen import tcmbench_datasets
    from opencompass.configs.models.TCMmodel.biancang import (models as biancang)
    from opencompass.configs.models.TCMmodel.huatuo import (models as huatuo)
    from opencompass.configs.models.TCMmodel.WiseDiag import (models as zzkj)
    from opencompass.configs.models.TCMmodel.spark_x1 import (models as spark)   
    from opencompass.configs.models.TCMmodel.baichuan3 import (models as baichuan)
    from opencompass.configs.models.TCMmodel.qwen_max import (models as qw)
    from opencompass.configs.models.TCMmodel.deepseek_v3 import (models as deepseek)
    from opencompass.configs.models.TCMmodel.gpt_35 import (models as gpt)
    from opencompass.configs.models.TCMmodel.llama3 import (models as llama)
    from opencompass.configs.models.TCMmodel.sunsimiao import (models as sunsimiao)
    from opencompass.configs.models.TCMmodel.qibo import (models as qibo)
    from opencompass.configs.models.TCMmodel.mingyi import (models as mingyi)
    from opencompass.configs.models.TCMmodel.doubao import (models as doubao)
    from opencompass.configs.models.TCMmodel.zhipu import (models as zhipu)
    from opencompass.configs.models.TCMmodel.gemini import (models as gemini)
    from opencompass.configs.models.TCMmodel.claude import (models as claude)
    from opencompass.configs.models.TCMmodel.zhongjing import (models as zhongjing)
    from opencompass.configs.models.TCMmodel.deepseek_r1 import (models as deepseek_r1)
    from opencompass.configs.datasets.TCMBench.tcm_llmjudge import (tcmbench_datasets,)
#opencompass examples/eval_tcmsubjective.py 
#from opencompass.models import (HuggingFace, HuggingFaceCausalLM, HuggingFaceChatGLM3, OpenAI)
from opencompass.partitioners import NaivePartitioner, SizePartitioner
from opencompass.partitioners.sub_naive import SubjectiveNaivePartitioner
from opencompass.partitioners.sub_num_worker import SubjectiveNumWorkerPartitioner
from opencompass.partitioners.sub_size import SubjectiveSizePartitioner
from opencompass.runners import LocalRunner, SlurmSequentialRunner
from opencompass.summarizers import SubjectiveSummarizer
from opencompass.tasks import OpenICLInferTask
from opencompass.tasks.subjective_eval import SubjectiveEvalTask

# Ensure each model has an 'abbr' key
# *deepseek,*biancang,*huatuo,*zzkj,*spark,*baichuan,*qw,*gpt,*llama,*sunsimiao,*qibo,*mingyi,*doubao,*zhipu,*gemini,*claude,*zhongjing
for model in deepseek:
    if 'abbr' not in model:
        model['abbr'] = f"{model.get('path', 'default')}"
for model in biancang:
    if 'abbr' not in model:
        model['abbr'] = f"{model.get('path', 'default')}"
for model in huatuo:
    if 'abbr' not in model:
        model['abbr'] = f"{model.get('path', 'default')}"
for model in zzkj:
    if 'abbr' not in model:
        model['abbr'] = f"{model.get('path', 'default')}"
for model in spark:
    if 'abbr' not in model:
        model['abbr'] = f"{model.get('path', 'default')}"
for model in baichuan:
    if 'abbr' not in model:
        model['abbr'] = f"{model.get('path', 'default')}"
for model in qw:
    if 'abbr' not in model:
        model['abbr'] = f"{model.get('path', 'default')}"
for model in gpt:
    if 'abbr' not in model:
        model['abbr'] = f"{model.get('path', 'default')}"
for model in llama:
    if 'abbr' not in model:
        model['abbr'] = f"{model.get('path', 'default')}"
for model in sunsimiao :
    if 'abbr' not in model:
        model['abbr'] = f"{model.get('path', 'default')}"
for model in qibo:
    if 'abbr' not in model:
        model['abbr'] = f"{model.get('path', 'default')}"
for model in mingyi:
    if 'abbr' not in model:
        model['abbr'] = f"{model.get('path', 'default')}"
for model in doubao:
    if 'abbr' not in model:
        model['abbr'] = f"{model.get('path', 'default')}"
for model in zhipu:
    if 'abbr' not in model:
        model['abbr'] = f"{model.get('path', 'default')}"
for model in gemini:
    if 'abbr' not in model:
        model['abbr'] = f"{model.get('path', 'default')}"
for model in claude:
    if 'abbr' not in model:
        model['abbr'] = f"{model.get('path', 'default')}"
for model in zhongjing:
    if 'abbr' not in model:
        model['abbr'] = f"{model.get('path', 'default')}"

api_meta_template = dict(round=[
    dict(role='HUMAN', api_role='HUMAN'),
    dict(role='BOT', api_role='BOT', generate=True),
])


# -------------Inference Stage ----------------------------------------

models=[*zzkj]
#models=[*baichuan]
#models=[*zhipu]
#models=[*qw]
#models=[*deepseek]
#models=[*claude]
#models=[*gemini]
#models=[*huatuo,*mingyi,*sunsimiao,*zhongjing,*baincang]

datasets = [*tcmbench_datasets]

infer = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(type=LocalRunner,
                max_num_workers=16,
                task=dict(type=OpenICLInferTask)),
            
)

# -------------Evalation Stage ----------------------------------------

## ------------- JudgeLLM Configuration
 # 确保 OpenAISDK 已被导入

# 设置 DeepSeek API 的 URL 和 API Key


judge_models = [
    *deepseek_r1,
]


## ------------- Evaluation Configuration
eval = dict(
    partitioner=dict(
        type=SubjectiveNaivePartitioner,
        models=models,
        judge_models=judge_models,
    ),
    runner=dict(type=LocalRunner,
                max_num_workers=16,
                task=dict(type=SubjectiveEvalTask)),
)

summarizer = dict(type=SubjectiveSummarizer, function='subjective')
work_dir = 'outputs/subjective/zhipu'
