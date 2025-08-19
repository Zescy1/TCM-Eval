import re
import csv
import json
from collections import defaultdict
import os

from opencompass.utils import model_abbr_from_cfg
from .utils import get_outdir

# 假设这些是已定义的常量和函数
All_Dimensions = ["accuracy", "completeness", "clarity", "conciseness", "relevance", "professionalism"]

def extract_rating(text):
    pattern = r'(\d+)\.\s*\*\*(.*?)\*\*：评分\s+(\d+)'
    matches = re.findall(pattern, text, re.IGNORECASE)
    print(f"Input Text: {text}")  # 调试信息
    print(f"Matches found: {matches}")  # 调试信息
    
    if not matches:
        print("No matches found in the input text.")  # 调试信息
        return {}
    
    values = [int(value) for _, _, value in matches]  # 修正了这里的逗号
    print(f"Extracted values: {values}")  # 调试信息
    
    # 提取前6个数字（索引为0到5）
    selected_values = values[:6]
    print(f"Selected values: {selected_values}")  # 调试信息
    
    result_dict = {}
    for i, dim in enumerate(All_Dimensions):
        if i < len(selected_values):
            result_dict[dim] = selected_values[i]
    
    print(f"Extracted Ratings: {result_dict}")  # 调试信息
    return result_dict

def post_process(judgement: str):
    """后处理函数：提取评分"""
    judgement = judgement.replace('\n', ' ')
    rating = extract_rating(judgement)
    #print(f"Extracted Ratings: {rating}")  # 调试信息
    return rating

def get_dimension_results(qa_data, fout, fout_flag):
    dimension_totals = defaultdict(float)
    dimension_counts = defaultdict(int)
    overall_weighted_sum = 0.0
    num_models = 0

    all_entries = list(qa_data.values())

    with open(fout, 'a+', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        if fout_flag == 0:
            header = ['模型'] + All_Dimensions + ['加权综合得分']
            writer.writerow(header)

        for entry in all_entries:
            judged_answers = post_process(entry.get("prediction", ""))
            
            if not judged_answers:
                print(f"No valid ratings extracted from entry: {entry}")
                continue
            
            model_name = entry.get("model_name", "Unknown Model")
            num_models += 1
            
            avg_row = [model_name]
            weighted_sum = 0.0
            
            for dim in All_Dimensions:
                score = judged_answers.get(dim, 0.0)
                dimension_totals[dim] += score
                dimension_counts[dim] += 1
                avg_row.append(score)
                
                # 计算当前模型的加权综合得分
                weights = {
                    "accuracy": 0.2,
                    "completeness": 0.2,
                    "clarity": 0.2,
                    "conciseness": 0.1,
                    "relevance": 0.15,
                    "professionalism": 0.15
                }
                weighted_sum += score * weights[dim]
            
            weighted_sum = round(weighted_sum, 2)
            avg_row.append(weighted_sum)
            overall_weighted_sum += weighted_sum
            
     #       print(f"Writing Row: {avg_row}")  # 调试信息
            writer.writerow(avg_row)
        
        if num_models > 0:
            overall_weighted_sum /= num_models
            overall_weighted_sum = round(overall_weighted_sum, 2)
            
            overall_row = ["Overall"]
            for dim in All_Dimensions:
                avg_score = dimension_totals[dim] / dimension_counts[dim] if dimension_counts[dim] > 0 else 0.0
                avg_score = round(avg_score, 2)
                overall_row.append(avg_score)
            overall_row.append(overall_weighted_sum)
            print(f"Writing Overall Row: {overall_row}")  # 调试信息
            writer.writerow(overall_row)

class TCMBenchSummarizer:

    def __init__(self, config: dict, judge_type='general') -> None:
        self.config = config
        self.eval_model_cfgs = self.config['eval']['partitioner']['models']
        self.eval_model_abbrs = [
            model_abbr_from_cfg(model) for model in self.eval_model_cfgs
        ]
        self.judge_models = self.config.get('judge_models', None)

    def summarize(self, time_str):
        for judge_model in self.judge_models:
            judge_abbr = model_abbr_from_cfg(judge_model)
            output_dir, results_folder = get_outdir(self.config, time_str)
            fout = os.path.join(output_dir, f'tcm_summary_{time_str}.csv')
            fout_flag = 0
            for eval_model_abbr in self.eval_model_abbrs:
                subdir = eval_model_abbr + '_judged-by--' + judge_abbr
                subdir_path = os.path.join(results_folder, subdir)
                dataset_cfgs = self.config['datasets']
                dataset = dataset_cfgs[0]  # Alignbench just have
                subdir_path = os.path.join(subdir_path, f"{dataset['name']}.json") 
                print(f"Reading QA data from: {subdir_path}")
                try:
                    with open(subdir_path, 'r', encoding='utf-8') as f:
                        qa_data = json.load(f)
                    print("QA data loaded successfully.")
                except FileNotFoundError:
                    print(f"Error: File not found - {subdir_path}")
                    continue  # 继续下一个文件而不是返回
                except json.JSONDecodeError:
                    print(f"Error: JSON decode error - {subdir_path}")
                    continue  # 继续下一个文件而不是返回
                print(f"Calculating dimension results and saving to: {fout}")
                get_dimension_results(qa_data, fout, fout_flag)
                fout_flag = 1  # 设置标志位以避免重复写入表头
                print(f"Summary saved to {fout}")
        return {'TCMbench': {}}



