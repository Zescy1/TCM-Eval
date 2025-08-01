from typing import List, Dict, Any, Set


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

class TCMBenchEvaluator:

    def score(self, predictions: List[List[str]], references: List[List[List[str]]]):
        """
        predictions: [[pred1], [pred2], ...] -> 每个 pred 是字符串
        references:  [[ans1], [ans2], ...] -> 每个 ans 是一个 List[str]
        """

        # 展平嵌套结构，提取最内层的关键词列表
        def extract_list(nested_list):
            while isinstance(nested_list, list) and len(nested_list) == 1:
                nested_list = nested_list[0]
            return nested_list

        # 处理预测值：从嵌套中提取字符串
        predictions = [extract_list(p) for p in predictions]
        # 处理参考答案：从嵌套中提取关键词列表
        references = [extract_list(r) for r in references]

        details = []
        total_score = 0          # 总共匹配的关键词数
        total_keywords = 0       # 所有题应匹配的关键词总数

        for pred, ref in zip(predictions, references):
            if not isinstance(ref, list):
                ref = [ref]  # 确保 ref 始终是列表
            parsed_pred = parse_key_multiple_answer(pred)  # list，不去重、不排序

            correct_answers = []
            incorrect_answers = []

            # 对每个标准答案关键词判断是否命中
            for keyword in ref:
                if keyword in parsed_pred:
                    correct_answers.append(keyword)
                else:
                    incorrect_answers.append(keyword)

            detail = {
                'pred': parsed_pred,
                'answer': ref,
                'correct': correct_answers,
                'incorrect': incorrect_answers,
                'score': len(correct_answers)  # 匹配几个得几分
            }

            total_score += len(correct_answers)
            total_keywords += len(ref)
            details.append(detail)

        accuracy = (total_score / total_keywords * 100) if total_keywords > 0 else 0
        return {
            'Total Matched': total_score,
            'Total Keywords': total_keywords,
            'Accuracy': round(accuracy, 2),
            'details': details
        }

# ======================
#   示例测试用例
# ======================

if __name__ == '__main__':
    evaluator = TCMBenchEvaluator()

    predictions = [
        ["医生通过望诊观察患者，再通过问诊了解病史，最后用切脉诊断"],
        ["患者自述腰痛，得温则减，医生闻其小便频数清长，切其脉象沉迟"],
        ["牙龈红肿溃烂、口臭明显，询问其牙痛剧烈，切其脉象洪数"],
        ["通过脉象浮数判断为肺热喘证"],
        ["无相关描述"],
        ["医生通过望诊和切诊进行了诊断"],
    ]

    references = [
        [["望", "问", "切"]],
        [["问", "闻", "切"]],
        [["望", "问", "切"]],
        [["切"]],
        [["望"]],
        [["望", "切"]],
    ]

    result = evaluator.score([[p] for p in predictions], references)

    print("总准确率:", result['Accuracy'], "%")
    print("\n每道题详细评分:")
    for i, detail in enumerate(result['details']):
        print(f"第 {i + 1} 题:")
        print("  预测内容:", detail['pred'])
        print("  正确答案:", detail['answer'])
        print("  匹配到的关键词:", detail['correct'])
        print("  错误关键词:", detail['incorrect'])
        print("  得分:", detail['score'], "%")
        print("-" * 50)