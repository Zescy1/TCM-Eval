from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import TCMBenchDataset,TCMBenchSinEvaluator,TCMBenchEvaluator_FOUR

tcmbench_reader_cfg = dict(
    input_columns=['problem_input'], output_column='label')
tcmbench_four_sets=['demo-b']
tcmbench_multiple_choices_sets = ['demo-a']  
tcmbench_datasets = []

for name in  tcmbench_multiple_choices_sets:
    tcmbench_infer_cfg = dict(
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(
                round=[dict(role='HUMAN', prompt='请解答（此题为单选题，请只输出正确选项）：{problem_input}\n')])),
        retriever=dict(type=ZeroRetriever
                       ),  
        inferencer=dict(type=GenInferencer))

    tcmbench_eval_cfg = dict(
        evaluator=dict(type=TCMBenchSinEvaluator), pred_role='BOT')

    tcmbench_datasets.append(
        dict(
            type=TCMBenchDataset,
            path='/home/opencompass/data/TCMbench/CH/',
            name=name,
            abbr=name,
            setting_name='zero-shot',
            reader_cfg=tcmbench_reader_cfg,
            infer_cfg=tcmbench_infer_cfg.copy(),
            eval_cfg=tcmbench_eval_cfg.copy()))
for name in  tcmbench_four_sets:
    tcmbench_infer_cfg = dict(
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(
                round=[dict(role='HUMAN', prompt='此题为四诊法填空题,只需要输出填空答案：{problem_input}\n')])),
        retriever=dict(type=ZeroRetriever
                       ),  
        inferencer=dict(type=GenInferencer))

    tcmbench_eval_cfg = dict(
        evaluator=dict(type=TCMBenchEvaluator_FOUR), pred_role='BOT')

    tcmbench_datasets.append(
        dict(
            type=TCMBenchDataset,
            path='/home/opencompass/data/TCMbench/CLOZE',
            name=name,
            abbr=name,
            setting_name='zero-shot',
            reader_cfg=tcmbench_reader_cfg,
            infer_cfg=tcmbench_infer_cfg.copy(),
            eval_cfg=tcmbench_eval_cfg.copy()))

del name, tcmbench_infer_cfg, tcmbench_eval_cfg
