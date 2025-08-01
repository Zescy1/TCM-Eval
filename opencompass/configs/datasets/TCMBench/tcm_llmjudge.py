from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.openicl.icl_evaluator import LMEvaluator
from opencompass.datasets import TCMBenchDataset,TCMBenchEvaluator_NLG
from opencompass.summarizers import TCMBenchSummarizer

tcmbench_reader_cfg = dict(
    input_columns=['problem_input'], output_column='label')

tcmbench_qa_sets = ['struct'] # 开放式QA，有标答

#tcmbench_ie_sets = ['DBMHG', 'CMeEE', 'CMeIE', 'CHIP-CDEE', 'CHIP-CDN', 'CHIP-CTC', 'Doc_parsing', 'IMCS-V2-MRG'] # 判断识别的实体是否一致，用F1评价

tcmbench_datasets = []

for name in tcmbench_qa_sets:
    tcmbench_infer_cfg = dict(
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(
                round=[dict(role='HUMAN', prompt='{problem_input}')])),
        retriever=dict(type=ZeroRetriever
                       ),  # retriver 不起作用，以输入参数为准 (zero-shot / few-shot)
        inferencer=dict(type=GenInferencer))

    #tcmbench_eval_cfg = dict(
      # evaluator=dict(type=TCMBenchEvaluator_NLG), pred_role='BOT')
    tcmbench_eval_cfg = dict(
        evaluator=dict(
            type=LMEvaluator,
            prompt_template=dict(
                type=PromptTemplate,
                template=dict(round=[
                    dict(
                        role='HUMAN',
                        prompt = '[问题]\n{problem_input}\n[助手的答案开始]\n{prediction}\n[助手的答案结束]\n' \
                    '[标准答案]{label}\n' \
                    '请给出每个维度的评分，范围是0-10，0表示不符合，10表示完全符合。请按照以下格式给出每个维度的评分。\n' \
                    '**评分格式**:\n' \
                    '1. **准确性 (Accuracy)**：评分 ，描述\n' \
                    '2. **完整性 (Completeness)**：评分  ，描述\n' \
                    '3. **清晰性 (Clarity)**：评分  ，描述\n' \
                    '4. **简洁性 (Conciseness)**：评分  ，描述\n' \
                    '5. **相关性 (Relevance)**：评分  ，描述\n' \
                    '6. **专业性 (Professionalism)**：评分  ，描述\n' \
                    '请严格按照上述格式给出评分。\n' \
                    '1."accuracy": "准确性：回答是否准确。"\n' \
                    '2."completeness": "完整性：涵盖关键点。"\n' \
                    '3."clarity": "清晰性：表达是否清楚。"\n' \
                    '4."conciseness": "简洁性：是否精炼。"\n' \
                    '5."relevance": "相关性：内容是否相关。"\n' \
                    '6."professionalism": "专业性：使用正确术语。"'
                   ),
                ]),
            ),
        ),
        pred_role='BOT',
    )
    tcmbench_datasets.append(
        dict(
            #abbr=f'{name}',
            type=TCMBenchDataset,
            path='/home/meihan.zhang/opencompass/data/TCMbench/STRUCT',#
            name=name,
            abbr=name,
            setting_name='zero-shot',
            reader_cfg=tcmbench_reader_cfg,
            infer_cfg=tcmbench_infer_cfg.copy(),
            eval_cfg=tcmbench_eval_cfg.copy(),
            mode='singlescore',
            summarizer = dict(type=TCMBenchSummarizer,judge_type='general', )
            ))


del name, tcmbench_infer_cfg, tcmbench_eval_cfg
