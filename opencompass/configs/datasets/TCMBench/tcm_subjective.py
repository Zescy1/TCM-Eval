from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import TCMBenchDataset, TCMBenchEvaluator_NLG, TCMBenchEvaluator,TCMBenchEvaluator_STR
tcmbench_reader_cfg = dict(
    input_columns=['problem_input'], output_column='label')

tcmbench_qa_sets = ['Open-Com','Pro-Herb','Reason','ShenNong']#,
tcmbench_cli_sets = ['Struct-Cli'] 
tcmbench_datasets = []
tcmbench_struct_cfg = dict(
    input_columns=['problem_input'], output_column=['label'])

for name in tcmbench_cli_sets:
    tcmbench_infer_cfg = dict(
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(
                round=[dict(role='HUMAN', prompt='请针对以下医案记录进行回答：{problem_input}。请回答这三个问题：1.用(证型:)的格式回答证型；2.用(方剂:)的格式回答方剂；3.用（中药:) 的格式回答中药（若有多个中药，用逗号分隔）。\n')])),
                #基于患者基本信息，进行回复：1.用(证型:)的格式回答证型；2.用(方剂:)的格式回答方剂；3.用（中药:) 的格式回答中药（若有多个中药，用逗号分隔）。{problem_input} \n
        retriever=dict(type=ZeroRetriever
                       ),  # retriver 不起作用，以输入参数为准 (zero-shot / few-shot)
        inferencer=dict(type=GenInferencer))
    tcmbench_eval_cfg = dict(
        evaluator=dict(type=TCMBenchEvaluator_NLG), pred_role='BOT')

    tcmbench_datasets.append(
        dict(
            type=TCMBenchDataset,
            path='/home/opencompass/data/TCMbench/STRUCT',
            name=name,
            abbr=name,
            setting_name='zero-shot',
            reader_cfg=tcmbench_reader_cfg,
            infer_cfg=tcmbench_infer_cfg.copy(),
            eval_cfg=tcmbench_eval_cfg.copy()))

for name in tcmbench_qa_sets:
    tcmbench_infer_cfg = dict(
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(
                round=[dict(role='HUMAN', prompt='请用中文回答问题:{problem_input}\n')])),
        retriever=dict(type=ZeroRetriever
                       ),  # retriver 不起作用，以输入参数为准 (zero-shot / few-shot)
        inferencer=dict(type=GenInferencer))

    tcmbench_eval_cfg = dict(
        evaluator=dict(type=TCMBenchEvaluator_NLG), pred_role='BOT')

    tcmbench_datasets.append(
        dict(
            type=TCMBenchDataset,
            path='/home/opencompass/data/TCMbench/QA/',
            name=name,
            abbr=name,
            setting_name='zero-shot',
            reader_cfg=tcmbench_reader_cfg,
            infer_cfg=tcmbench_infer_cfg.copy(),
            eval_cfg=tcmbench_eval_cfg.copy()))

del name, tcmbench_infer_cfg, tcmbench_eval_cfg
