# flake8: noqa
import json
import re
from typing import List, Dict, Any, Set
from . import dataset_loader
from difflib import SequenceMatcher

def extract_last_line(string):
    lines = string.split('\n')
    for item in lines[::-1]:
        if item.strip() != '':
            string = item
            break
    return string

def parse_key_multiple_answer(pred: str) -> List[str]:
    """
    从模型输出字符串中提取出“望”、“闻”、“问”、“切”等关键字。
    只要出现这些字，就认为对应关键词存在。
    示例：
        pred = "医生通过望诊观察患者，再通过问诊了解病史，最后用切脉诊断"
        返回 ["望", "问", "切"]
    """
    found = []

    if '望' in pred:
        found.append('望')
    if '闻' in pred:
        found.append('闻')
    if '问' in pred:
        found.append('问')
    if '切' in pred or '脉' in pred:
        found.append('切')

    return found


def remove_few_shot_prefix(string: str):
    prefix_list = ['The answer is therefore', '答案是']
    for prefix in prefix_list:
        if string.startswith(prefix):
            string = string[len(prefix):].strip()
        elif prefix in string:
            index = string.rfind(prefix)
            if index >= 0:
                string = string[index + len(prefix):].strip()
    return string


def try_parse_few_shot_qa_single_answer(string, setting_name, language='en'):
    if setting_name == 'few-shot-CoT':
        string = extract_last_line(string)
    if language == 'en':
        pattern = 'answer is .*?([A-G])'
        match = re.search(pattern, string)
    elif language == 'zh':
        pattern = '答案是.*?([A-G])'
        match = re.search(pattern, string)
    else:
        raise ValueError('Unknown language {0}'.format(language))
    if match:
        return match.group(1)
    else:
        return None


def try_parse_few_shot_pattern(string: str, dataset_name, setting_name):
    if setting_name == 'few-shot-CoT':
        string = extract_last_line(string)
    if dataset_name in dataset_loader.chinese_cloze_datasets:
        return string.startswith('答案是')
    elif dataset_name in dataset_loader.english_cloze_datasets:
        return string.startswith('The answer is therefore')
    elif dataset_name in dataset_loader.chinese_qa_datasets:
        pattern = '答案是.*?([A-G])'
        match = re.search(pattern, string)
        return match is not None
    elif dataset_name in dataset_loader.english_qa_datasets:
        pattern = 'answer is .*?([A-G])'
        match = re.search(pattern, string)
        return match is not None
    return False


def parse_few_shot_qa_single_answer(string, setting_name, language='en'):
    answer = try_parse_few_shot_qa_single_answer(string, setting_name,
                                                 language)
    if answer is None:
        return find_first_capital_letter(string)
    else:
        return answer


def find_first_capital_letter(answer):
    letter_set = {'A', 'B', 'C', 'D', 'E', 'F'}
    for c in answer:
        if c in letter_set:
            return c
    # print("Can't find capital letter in:", answer)
    return ''


def extract_answer_in_bracket(answer, prefix='“', suffix='”'):
    if prefix not in answer and suffix not in answer:
        # print("doesn't found special tokens in:", answer)
        return ''
    s = answer.index(prefix) + len(prefix)
    t = answer.index(suffix)
    ret = answer[s:t]
    return ret


def parse_math_answer(setting_name, raw_string):
    if setting_name == 'few-shot-CoT':
        raw_string = extract_last_line(raw_string)
    if setting_name == 'few-shot-CoT' or setting_name == 'few-shot':
        raw_string = remove_few_shot_prefix(raw_string)
        return raw_string

    def remove_boxed(s):
        left = '\\boxed{'
        try:
            assert s[:len(left)] == left
            assert s[-1] == '}'
            answer = s[len(left):-1]
            if '=' in answer:
                answer = answer.split('=')[-1].lstrip(' ')
            return answer
        except:
            return None

    def last_boxed_only_string(string):
        idx = string.rfind('\\boxed')
        if idx < 0:
            idx = string.rfind('\\fbox')
            if idx < 0:
                return None
        i = idx
        right_brace_idx = None
        num_left_braces_open = 0
        while i < len(string):
            if string[i] == '{':
                num_left_braces_open += 1
            if string[i] == '}':
                num_left_braces_open -= 1
                if num_left_braces_open == 0:
                    right_brace_idx = i
                    break
            i += 1

        if right_brace_idx == None:
            retval = None
        else:
            retval = string[idx:right_brace_idx + 1]

        return retval

    def get_answer_with_dollar_sign(s):
        first_pattern = '\$(.*)\$'
        last_match = None
        matches = re.findall(first_pattern, s)
        if matches:
            last_match = matches[-1]
            if '=' in last_match:
                last_match = last_match.split('=')[-1].lstrip(' ')
        return last_match

    def get_answer_without_dollar_sign(s):
        last_match = None
        if '=' in s:
            last_match = s.split('=')[-1].lstrip(' ').rstrip('.')
            if '\n' in last_match:
                last_match = last_match.split('\n')[0]
        else:
            pattern = '(?:\\$)?\d+(?:\.\d+)?(?![\w\d])'
            matches = re.findall(pattern, s)
            if matches:
                last_match = matches[-1]
        return last_match

    raw_string = remove_few_shot_prefix(raw_string)
    if '\\boxed' in raw_string:
        answer = remove_boxed(last_boxed_only_string(raw_string))
    else:
        answer = get_answer_with_dollar_sign(raw_string)
        if not answer:
            answer = get_answer_without_dollar_sign(raw_string)
    return answer
import re

def strcut_extract(string, return_dict=True):
    """
    从字符串中提取 '证型'、'方剂'、'中药' 字段内容。
    
    参数:
        string (str): 输入文本（模型输出或标准答案）
        return_dict (bool): 是否返回字典，默认为 True，否则返回 list
    
    返回:
        dict or list: 包含提取字段的结构化数据
    """
    # 正则模式定义
    pattern_zhengxing = r'证型[:：]?(.*?)[,，]'
    pattern_fangji = r'方剂[:：]?(.*?)[,，]'
    pattern_yaocai = r'中药[:：]?(.*?)[,，]'

    # 提取信息
    zhengxing = re.findall(pattern_zhengxing, string)
    fangji = re.findall(pattern_fangji, string)
    yaocai = re.findall(pattern_yaocai, string)

    # 取第一个匹配项并去除前后空格
    zhengxing = zhengxing[0].strip() if zhengxing else ""
    fangji = fangji[0].strip() if fangji else ""
    yaocai = yaocai[0].strip() if yaocai else ""

    # 返回结构化结果
    if return_dict:
        return {
            "证型": zhengxing,
            "方剂": fangji,
            "中药": yaocai
        }
    else:
        return [zhengxing, fangji, yaocai]
from difflib import SequenceMatcher

def is_similar(a, b, threshold=0.8):
    """判断两个字符串是否足够相似"""
    if not a or not b:
        return False
    ratio = SequenceMatcher(None, a, b).ratio()
    return ratio >= threshold

def cloze_book(string):
    # 使用正则表达式匹配书名号及其内部的内容
    pattern = r'《(.*?)》'
    # 找到所有匹配项
    matches = re.findall(pattern, string)
    # 返回匹配结果列表
    return [f'《{title}》' for title in matches]



def parse_qa_multiple_answer(string):
    """
    从任意文本中提取所有 A-F 字母（不区分大小写），用于选择题答案解析。
    
    支持格式：
        - 答案是B.乡村医生...
        - 【答案】A.市场上有很多供应...
        - ABDE
        - (AC)
        - [BC]
        - A,B,C,D
        
    参数:
        string (str): 输入文本
    
    返回:
        List[str]: 提取到的答案字母列表，如 ['A', 'C']
    """
    if not isinstance(string, str):
        return []

    # 找出所有 A-F（不区分大小写）的字母
    matches = re.findall(r'[A-F]', string)

    # 去重 + 按照首次出现顺序保留
    seen = set()
    result = []
    for letter in matches:
        upper = letter.upper()
        if upper not in seen:
            seen.add(upper)
            result.append(upper)

    return result


def post_process(dataset_name, setting_name, prediction):
    if dataset_name in dataset_loader.english_cloze_datasets or dataset_name in dataset_loader.chinese_cloze_datasets:
        return parse_math_answer(setting_name, prediction)

    if dataset_name in ['jec-qa-kd', 'jec-qa-ca', 'gaokao-physics']:
        return parse_qa_multiple_answer(prediction, setting_name)

    # all other datasets are QA problems with single answer
    if 'zero-shot' in setting_name:
        answer = find_first_capital_letter(prediction)
        return answer

    # all other datasets are QA problems with single answer and setting_name are few-shot
    language = 'en' if dataset_name in dataset_loader.english_qa_datasets else 'zh'
    if dataset_name in dataset_loader.english_qa_datasets or dataset_name in dataset_loader.chinese_qa_datasets:
        return parse_few_shot_qa_single_answer(prediction, setting_name,
                                               language)
    else:
        raise ValueError(f'Unsupported dataset name {dataset_name}')
