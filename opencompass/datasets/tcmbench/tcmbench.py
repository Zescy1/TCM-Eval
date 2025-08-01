import json
import os.path as osp
import sys
from datasets import Dataset
from sklearn.metrics import classification_report
from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS, LOAD_DATASET
from opencompass.utils import get_data_path

from opencompass.datasets.base import BaseDataset
from opencompass.datasets.tcmbench.math_equivalence import is_equiv
from opencompass.datasets.tcmbench.post_process import  extract_answer_in_bracket,parse_key_multiple_answer, parse_qa_multiple_answer,cloze_book,strcut_extract,is_similar  #,parse_math_answer
from opencompass.datasets.tcmbench.openqa import calc_scores_nlg

@ICL_EVALUATORS.register_module()
class TCMBenchEvaluator_NLG(BaseEvaluator):

    def score(self, predictions, references):
        return calc_scores_nlg(predictions, references)

@ICL_EVALUATORS.register_module()
class TCMBenchEvaluator_CLI(BaseEvaluator):
    def score(self, predictions, references):
        # predictions: [[]]
        # references: [[]]
        predictions = [extract_answer_in_bracket(pred) for pred in predictions]
        details = []
        cnt = 0
        for pred, ref in zip(predictions, references):
            detail = {'pred': pred, 'answer': ref, 'correct': False}
            if is_equiv(pred, ref):
                cnt += 1
                detail['correct'] = True
            details.append(detail)
        score = cnt / len(predictions) * 100
        return {'Accuracy': score, 'details': details}
    
@ICL_EVALUATORS.register_module()
class TCMBenchEvaluator_FOUR(BaseEvaluator):

    def score(self, predictions, references):
        """
        predictions: [[pred1], [pred2], ...] -> 每个 pred 是字符串
        references:  [[ans1], [ans2], ...] -> 每个 ans 是一个 List[str]
        """

        # 展平嵌套结构，提取最内层的关键词列表
        def extract_list(nested_list):
            while isinstance(nested_list, list) and len(nested_list) == 1:
                nested_list = nested_list[0]
            return nested_list

        predictions = [extract_list(p) for p in predictions]
        references = [extract_list(r) for r in references]

        total_score = 0
        total_keywords = 0
        details = []

        for idx, (pred, ref) in enumerate(zip(predictions, references)):
            if not isinstance(ref, list):
                ref = [ref]

            parsed_pred = parse_key_multiple_answer(pred)

            correct_answers = [k for k in parsed_pred if k in ref]
            score = len(correct_answers)

            detail = {
                'pred': parsed_pred,
                'answer': ref,
                'score': score
            }

            total_score += score
            total_keywords += len(ref)
            details.append(detail)

        accuracy = (total_score / total_keywords * 100) if total_keywords > 0 else 0

        return {
            'Accuracy': round(accuracy, 2),
            'details': details
        }
    
@ICL_EVALUATORS.register_module()
class TCMBenchEvaluator_STR(BaseEvaluator):
    def score(self, predictions, references):
        """
        对预测结果和参考答案进行评分。
        使用 strcut_extract 提取结构化字段，并逐项比对。
        Cloze 分改为基于每个字段的 NLG 指标打分（调用 calc_nlg_task_scores）。
        """

        details = []
        total_score = 0
        field_weights = {
            "证型": 5,
            "方剂": 5,
            "中药": 5,
        }
        fields = list(field_weights.keys())

        for idx, (pred, ref) in enumerate(zip(predictions, references)):
            detail = {
                'question_index': idx + 1,
                'pred': pred,
                'answer': ref,
                'cloze': 0,
            }

            # 使用 strcut_extract 提取结构化字段
            pred_dict = strcut_extract(pred)
            ref_dict = strcut_extract(ref)

            # Cloze 分改为基于每个字段的 NLG 打分
            cloze_score = 0
            field_scores = {}

            for field in fields:
                pred_val = pred_dict[field].strip()
                ref_val = ref_dict[field].strip()

                if not pred_val or not ref_val:
                    field_scores[field] = 0
                    continue

                # 调用你已有的 calc_nlg_task_scores 函数，传入两个字符串
                result = calc_scores_nlg([pred_val], [ref_val])
                rouge_l_f1 = result['BertScore_F1_Avg']
                field_score = round(rouge_l_f1 * field_weights[field], 2)
                field_score = min(field_score, field_weights[field])  # 不超过满分
                field_scores[field] = field_score
                cloze_score += field_score

            detail['cloze'] = round(cloze_score, 2)


            detail['total_score'] = round(cloze_score/0.15, 2)
            
            total_score += cloze_score/0.15
            details.append(detail)

        # 计算平均得分（百分制）
        avg_score = round(total_score / len(predictions), 2)

        return {
            'Score': avg_score,
            'details': details
        }
    
@LOAD_DATASET.register_module()
class TCMBenchDataset(BaseDataset):

    @staticmethod
    def load(path: str, name: str, setting_name: str,*args,
             **kwargs):
        path = get_data_path(path, local_mode=True)
        from .dataset_loader import load_dataset, load_dataset_as_result_schema

        assert setting_name in 'zero-shot', 'only support zero-shot setting'
        dataset_wo_label = load_dataset(name,  path)
        dataset_with_label = load_dataset_as_result_schema(name, path)
        dataset = []
        for d1, d2 in zip(dataset_wo_label, dataset_with_label):
            dataset.append({
                'id': d2.get('index'),           # 修改为 dict 取值方式
                'problem_input': d1.get('context'),
                'label': d2.get('label') 
            })
        dataset = Dataset.from_list(dataset)
        return dataset

@ICL_EVALUATORS.register_module()
class TCMBenchEvaluator_Syn(BaseEvaluator):

    def score(self, predictions, references):
        details = []
        cnt = 0

        for pred, ref in zip(predictions, references):
            detail = {'pred': pred, 'answer': ref, 'correct': False}
            
            # Check if all symptoms in ref are present in pred
            if all(item in pred for item in ref):
                cnt += 1
                detail['correct'] = True
            
            details.append(detail)
        
        score = (cnt / len(predictions)) * 100 if predictions else 0
        return {'Accuracy': score, 'details': details}

#计算症状
@ICL_EVALUATORS.register_module()
class TCMBenchEvaluator_Book(BaseEvaluator):

    def score(self, predictions, references):
        details = []
        cnt = 0

        for pred, ref in zip(predictions, references):
            preds = cloze_book(pred)
            refs = cloze_book(ref)
            detail = {'pred':preds, 'answer':refs, 'correct':False}
            if any(item in preds for item in refs):
                cnt += 1
                detail['correct'] = True
            details.append(detail)
        score = cnt / len(predictions) * 100
        return {'Accuracy': score, 'details': details}
### 纯中文判断题，不能用TF替代
@ICL_EVALUATORS.register_module()
class TCMBenchEvaluator_TF(BaseEvaluator):

    def score(self, predictions, references):
        details = []
        cnt = 0

        for pred, ref in zip(predictions, references):
            
            if '不正确' in pred or 'F' in pred:
                cur_pred = ['F']
            else:
                cur_pred = ['T']

            detail = {'pred':cur_pred, 'answer':ref, 'correct':False}

            if cur_pred == ref:
                cnt += 1
                detail['correct'] = True
            
            details.append(detail)

        score = cnt / len(predictions) * 100
        return {'Accuracy': score, 'details': details}
    
 #客观题其实没差别   
@LOAD_DATASET.register_module()
class TCMSumDataset(BaseDataset):

    @staticmethod
    def load(path: str, name: str, setting_name: str,*args,
             **kwargs):
        path = get_data_path(path, local_mode=True)
        from .dataset_loader import load_dataset, load_dataset_as_result_schema

        assert setting_name in 'zero-shot', 'only support zero-shot setting'
        dataset_wo_label = load_dataset(name, setting_name, path)
        dataset_with_label = load_dataset_as_result_schema(name, path)
        dataset = []
        for d1, d2 in zip(dataset_wo_label, dataset_with_label):
            dataset.append({
                'id': d2.index,
                'problem_input': d1['context'],
                'label': d2.label,
            })
        dataset = Dataset.from_list(dataset)
        return dataset

@ICL_EVALUATORS.register_module()
class TCMBenchEvaluator(BaseEvaluator):

    def score(self, predictions, references):
        # predictions: [[]]
        # references: [[]]
        predictions = [parse_qa_multiple_answer(pred) for pred in predictions]
        references = [list(ref) for ref in references]  # 确保也都是 list 形式
        details = []
        cnt = 0
        for pred, ref in zip(predictions, references):
            detail = {'pred': pred, 'answer': ref, 'correct': False}
            if is_equiv(pred, ref):
                cnt += 1
                detail['correct'] = True
            details.append(detail)
        score = cnt / len(predictions) * 100
        return {'Accuracy': score, 'details': details}
    
@ICL_EVALUATORS.register_module()
class TCMBenchSinEvaluator(BaseEvaluator):

    def score(self, predictions, references):
        # predictions: [[]]
        # references: [[]]
        predictions = [parse_qa_multiple_answer(pred)[:1] if pred else [] for pred in predictions]#
        details = []
        cnt = 0
        for pred, ref in zip(predictions, references):
            detail = {'pred': pred, 'answer': ref, 'correct': False}
            if is_equiv(pred, ref):
                cnt += 1
                detail['correct'] = True
            details.append(detail)
        score = cnt / len(predictions) * 100
        return {'Accuracy': score, 'details': details}