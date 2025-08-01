from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import TCMBenchDataset, TCMBenchEvaluator_Book,TCMBenchEvaluator, TCMBenchEvaluator_TF,TCMBenchSinEvaluator
from opencompass.utils.text_postprocessors import first_capital_postprocess

tcmbench_reader_cfg = dict(
    input_columns=['problem_input'], output_column='label')
tcmbench_four_sets=['four']
tcmbench_cli_sets=['cli']
tcmbench_book_sets = ['book'] #gpt-4 
tcmbench_multiple_multichoices_sets = ['work-much','law-much'] # 选择题，用acc判断
tcmbench_multiple_choices_sets = ['base-multi','knowledge-multi','law-multi','logic-multi','quanlity-multi','sj-multi','work-multi'] # 选择题，用acc判断   
tcmbench_depart_sets=['appear','bones','child','inner','needles','outter','recover','tuina','woman']
tcmbench_depart_sets=['try-b']
tcmbench_single_choice_sets = ['base-single','knowledge-single','logic-single','sj-single'] # 正确与否判断，有标答

tcmbench_datasets = []

for name in  tcmbench_depart_sets:
    tcmbench_infer_cfg = dict(
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(
                round=[dict(role='HUMAN', prompt='请解答（此题为单选题，请只输出正确选项）：{problem_input}\n')])),
        retriever=dict(type=ZeroRetriever
                       ),  # retriver 不起作用，以输入参数为准 (zero-shot / few-shot)
        inferencer=dict(type=GenInferencer))

    tcmbench_eval_cfg = dict(
        evaluator=dict(type=TCMBenchSinEvaluator), pred_role='BOT')

    tcmbench_datasets.append(
        dict(
            type=TCMBenchDataset,
            path='/home/meihan.zhang/opencompass/data/TCMbench/DEPART',
            name=name,
            abbr=name,
            setting_name='zero-shot',
            reader_cfg=tcmbench_reader_cfg,
            infer_cfg=tcmbench_infer_cfg.copy(),
            eval_cfg=tcmbench_eval_cfg.copy()))
'''
for name in  tcmbench_multiple_choices_sets:
    tcmbench_infer_cfg = dict(
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(
                round=[dict(role='HUMAN', prompt='请解答（此题为单选题，请只输出正确选项）：{problem_input}\n')])),
        retriever=dict(type=ZeroRetriever
                       ),  # retriver 不起作用，以输入参数为准 (zero-shot / few-shot)
        inferencer=dict(type=GenInferencer))

    tcmbench_eval_cfg = dict(
        evaluator=dict(type=TCMBenchSinEvaluator), pred_role='BOT')

    tcmbench_datasets.append(
        dict(
            type=TCMBenchDataset,
            path='/home/meihan.zhang/opencompass/data/TCMbench/CH/',
            name=name,
            abbr=name,
            setting_name='zero-shot',
            reader_cfg=tcmbench_reader_cfg,
            infer_cfg=tcmbench_infer_cfg.copy(),
            eval_cfg=tcmbench_eval_cfg.copy()))

for name in tcmbench_single_choice_sets:
    tcmbench_infer_cfg = dict(
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(
                round=[dict(role='HUMAN', prompt='请解答：{problem_input}（请用正确T或者错误F回答问题）\n')])),
        retriever=dict(type=ZeroRetriever
                       ),  # retriver 不起作用，以输入参数为准 (zero-shot / few-shot)
        inferencer=dict(type=GenInferencer))

    tcmbench_eval_cfg = dict(
        evaluator=dict(type=TCMBenchEvaluator_TF), pred_role='BOT')

    tcmbench_datasets.append(
        dict(
            type=TCMBenchDataset,
            path='/home/meihan.zhang/opencompass/data/TCMbench/TF/',#TF/' ,
            name=name,
            abbr=name,
            setting_name='zero-shot',
            reader_cfg=tcmbench_reader_cfg,
            infer_cfg=tcmbench_infer_cfg.copy(),
            eval_cfg=tcmbench_eval_cfg.copy()))

for name in tcmbench_multiple_multichoices_sets:
    tcmbench_infer_cfg = dict(
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(
                round=[dict(role='HUMAN', prompt='请解答（此题为多选，请只输出正确答案）：{problem_input}\n')])),
        retriever=dict(type=ZeroRetriever
                       ),  # retriver 不起作用，以输入参数为准 (zero-shot / few-shot)
        inferencer=dict(type=GenInferencer))

    tcmbench_eval_cfg = dict(
        evaluator=dict(type=TCMBenchEvaluator), pred_role='BOT')

    tcmbench_datasets.append(
        dict(
            type=TCMBenchDataset,
            path='/home/meihan.zhang/opencompass/data/TCMbench/MUCH/' ,
            name=name,
            abbr=name,
            setting_name='zero-shot',
            reader_cfg=tcmbench_reader_cfg,
            infer_cfg=tcmbench_infer_cfg.copy(),
            eval_cfg=tcmbench_eval_cfg.copy()))

for name in tcmbench_book_sets:
    tcmbench_infer_cfg = dict(
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(
                round=[dict(role='HUMAN', prompt='请只用中医药的书籍回答问题：{problem_input}\n')])),
        retriever=dict(type=ZeroRetriever
                       ),  # retriver 不起作用，以输入参数为准 (zero-shot / few-shot)
        inferencer=dict(type=GenInferencer))

    tcmbench_eval_cfg = dict(
        evaluator=dict(type=TCMBenchEvaluator_Book), pred_role='BOT')

    tcmbench_datasets.append(
        dict(
            type=TCMBenchDataset,
            path='/home/meihan.zhang/opencompass/data/TCMbench/CLOZE',#TF/' ,
            name=name,
            abbr=name,
            setting_name='zero-shot',
            reader_cfg=tcmbench_reader_cfg,
            infer_cfg=tcmbench_infer_cfg.copy(),
            eval_cfg=tcmbench_eval_cfg.copy()))
'''
del name, tcmbench_infer_cfg, tcmbench_eval_cfg
