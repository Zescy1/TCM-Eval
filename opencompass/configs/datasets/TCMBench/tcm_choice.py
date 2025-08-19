from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import TCMBenchDataset, TCMBenchEvaluator_Book,TCMBenchEvaluator, TCMBenchEvaluator_TF,TCMBenchSinEvaluator, TCMBenchEvaluator_FOUR,TCMBenchEvaluator_CLI

tcmbench_reader_cfg = dict(
    input_columns=['problem_input'], output_column='label')
tcmbench_four_sets=['Four']
tcmbench_cli_sets=['cli']
tcmbench_book_sets = ['Book'] #gpt-4 
tcmbench_multiple_multichoices_sets = ['Licen-MCH','Law-MCH'] 
tcmbench_multiple_choices_sets = ['Base-CH','Herb-CH','Law-CH','Logic-CH','Licen-CH','Prac-CH','TCM-CH']  
tcmbench_depart_sets=['Acup-CH','Derm-CH','Gyne-CH','Inter-CH','Orth-CH','Otorh-CH','Pedi-CH','Pehab-CH','Tuina-CH']
tcmbench_single_choice_sets = ['Base-TF','Herb-TF','Logic-TF','Prac-TF']

tcmbench_datasets = []

for name in  tcmbench_depart_sets:
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
            path='/home/opencompass/data/TCMbench/DEPART',
            name=name,
            abbr=name,
            setting_name='zero-shot',
            reader_cfg=tcmbench_reader_cfg,
            infer_cfg=tcmbench_infer_cfg.copy(),
            eval_cfg=tcmbench_eval_cfg.copy()))

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

for name in tcmbench_single_choice_sets:
    tcmbench_infer_cfg = dict(
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(
                round=[dict(role='HUMAN', prompt='请解答：{problem_input}（请用正确T或者错误F回答问题）\n')])),
        retriever=dict(type=ZeroRetriever
                       ),  
        inferencer=dict(type=GenInferencer))

    tcmbench_eval_cfg = dict(
        evaluator=dict(type=TCMBenchEvaluator_TF), pred_role='BOT')

    tcmbench_datasets.append(
        dict(
            type=TCMBenchDataset,
            path='/home/opencompass/data/TCMbench/TF/',
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
            path='/home/opencompass/data/TCMbench/MUCH/' ,
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
                       ),  
        inferencer=dict(type=GenInferencer))

    tcmbench_eval_cfg = dict(
        evaluator=dict(type=TCMBenchEvaluator_Book), pred_role='BOT')

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
'''
for name in  tcmbench_cli_sets:
    tcmbench_infer_cfg = dict(
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(
                round=[dict(role='HUMAN', prompt='此题为填空题,请用双引号“”输出每个空的答案,只给出答案即可：{problem_input}\n')])),
        retriever=dict(type=ZeroRetriever
                       ),  
        inferencer=dict(type=GenInferencer))

    tcmbench_eval_cfg = dict(
        evaluator=dict(type=TCMBenchEvaluator_CLI), pred_role='BOT')

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
'''
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
