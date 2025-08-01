# flake8: noqa
import ast
import json
import os

import pandas as pd
import tiktoken
from tqdm import tqdm

from .constructions import ChatGPTSchema, ResultsForHumanSchema
from .utils import extract_answer, read_jsonl, save_jsonl
# 定义开放式QA数据集
datasets = ['try','try-b','Process','Reason','shennong','SynDia','struct','four','cli','book',\
            'work-much','law-much','base-multi','knowledge-multi','law-multi','logic-multi','quanlity-multi','sj-multi','work-multi',\
                'base-single','knowledge-single','logic-single','sj-single','struct',\
                   'appear','bones','child','inner','needles','outter','recover','tuina','woman' ]


def convert_zero_shot(line, dataset_name):
    question = line.get('question')
    answer = line.get('answer')

    if not question:
        print("⚠️ 警告：缺少 'question' 字段", line)
    if not answer:
        print("⚠️ 警告：缺少 'answer' 字段", line)

    return {
        'question': question or '',
        'answer': answer or ''
    }

def load_dataset(dataset_name, parent_path):
    """加载并处理指定的数据集"""
    test_path = f"{parent_path}/{dataset_name}.jsonl"
    
    # 加载数据（假设是 JSON Lines 格式）
    processed = []
    if dataset_name in datasets:
        with open(test_path, 'r', encoding='utf-8') as file:
            for meta_idx, line in enumerate(tqdm(file, desc="Processing")):
                try:
                    data_line = json.loads(line.strip())  # 每行是一个 JSON 对象
                    processed_line = convert_zero_shot(data_line, dataset_name)
                    processed.append({
                        'context': processed_line['question'],  # query 转换为问题
                        'metadata': meta_idx,
                        'label': processed_line['answer']  # response 转换为答案
                    })
                except json.JSONDecodeError as e:
                    print(f"JSON 解码错误，跳过第 {meta_idx + 1} 行：{e}")
    else:
        raise ValueError(f"未知的数据集: {dataset_name}")
    return processed

def load_dataset_as_result_schema(dataset_name, parent_path):
    test_path = f"{parent_path}/{dataset_name}.jsonl"
    print("正在读取文件:", test_path)

    if not os.path.exists(test_path):
        raise FileNotFoundError(f"找不到文件: {test_path}")

    processed = []
    with open(test_path, 'r', encoding='utf-8') as file:
        for i, line in enumerate(file):
            try:
                data_line = json.loads(line.strip())
                #print(f"第{i}行原始数据:", data_line)  # 调试输出
                processed_line = convert_zero_shot(data_line, dataset_name)
                #print(f"第{i}行转换后数据:", processed_line)  # 调试输出
                processed.append({
                    'index': i,
                    'problem_input': processed_line.get('question'),
                    'label': processed_line.get('answer')
                })
            except json.JSONDecodeError as e:
                print(f"JSON 解码错误，跳过第 {i + 1} 行：{e}")
    
    print("最终生成的数据长度:", len(processed))  # 调试输出
    return processed
if __name__ == '__main__':
    parent_dir = '/home/meihan.zhang/opencompass/data/TCMbench/'
    data_name = 'ChatMed_temp'  # 设置为您的开放式QA数据集名称
    save_dir = '/home/meihan.zhang/opencompass/experiment_input/zero-shot/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    processed_data = load_dataset(data_name, parent_dir)
    with open(f"{save_dir}{data_name}.jsonl", 'w', encoding='utf-8') as outfile:
        for item in processed_data:
            json.dump(item, outfile, ensure_ascii=False)
            outfile.write('\n')

    human_readable_results = load_dataset_as_result_schema(data_name, parent_dir)
    with open(f"{save_dir}human_readable_{data_name}.json", 'w', encoding='utf-8') as outfile:
        for item in human_readable_results:
            json.dump(item, outfile, ensure_ascii=False)
            outfile.write('\n')