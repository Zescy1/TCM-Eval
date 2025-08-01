from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import TCMBenchDataset,  TCMBenchEvaluator_FOUR,TCMBenchEvaluator_CLI
from opencompass.utils.text_postprocessors import first_capital_postprocess

tcmbench_reader_cfg = dict(
    input_columns=['problem_input'], output_column='label')
tcmbench_four_sets=['four']

tcmbench_cli_sets=['cli']
tcmbench_datasets = []
for name in  tcmbench_cli_sets:
    tcmbench_infer_cfg = dict(
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(
                round=[dict(role='HUMAN', prompt='此题为填空题,请用双引号“”输出每个空的答案,只给出答案即可：{problem_input}\n')])),
        retriever=dict(type=ZeroRetriever
                       ),  # retriver 不起作用，以输入参数为准 (zero-shot / few-shot)
        inferencer=dict(type=GenInferencer))

    tcmbench_eval_cfg = dict(
        evaluator=dict(type=TCMBenchEvaluator_CLI), pred_role='BOT')

    tcmbench_datasets.append(
        dict(
            type=TCMBenchDataset,
            path='/home/meihan.zhang/opencompass/data/TCMbench/CLOZE',
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
                       ),  # retriver 不起作用，以输入参数为准 (zero-shot / few-shot)
        inferencer=dict(type=GenInferencer))

    tcmbench_eval_cfg = dict(
        evaluator=dict(type=TCMBenchEvaluator_FOUR), pred_role='BOT')

    tcmbench_datasets.append(
        dict(
            type=TCMBenchDataset,
            path='/home/meihan.zhang/opencompass/data/TCMbench/CLOZE',
            name=name,
            abbr=name,
            setting_name='zero-shot',
            reader_cfg=tcmbench_reader_cfg,
            infer_cfg=tcmbench_infer_cfg.copy(),
            eval_cfg=tcmbench_eval_cfg.copy()))

del name, tcmbench_infer_cfg, tcmbench_eval_cfg
