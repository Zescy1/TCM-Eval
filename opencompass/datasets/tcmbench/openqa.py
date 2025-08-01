from transformers import BasicTokenizer
from rouge_chinese import Rouge
from bert_score import score
import torch.nn.functional as F
import torch
from collections import Counter
import numpy as np
import jieba
import re
from sentence_transformers import SentenceTransformer, models

basic_tokenizer = BasicTokenizer(tokenize_chinese_chars=True)

# 使用你自己的模型路径
model1_path = "/home/meihan.zhang/models/sbert-base-chinese-nl/"
model2_path = "/home/meihan.zhang/models/bert-base-chinese"
tokenizer_path = model2_path

# 构建 SentenceTransformer 模型
word_embedding_model = models.Transformer(model1_path)
pooling_model = models.Pooling(
    word_embedding_model.get_word_embedding_dimension(),
    pooling_mode_mean_tokens=True,
    pooling_mode_cls_token=False,
    pooling_mode_max_tokens=False
)
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

def tokenize_chinese_chars(text):
    # 清洗非中文字符
    text = re.sub(r'[^\u4e00-\u9fa5\s]', '', text).strip()
    tokens = list(jieba.cut(text))
    
    # 简单的停用词过滤
    stopwords = set([
        '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一',
        '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '看',
        '他', '她', '它', '这', '那', '他们', '我们', '你们', '对', '为',
        '能', '可以', '没有', '中', '之', '与', '于', '后', '前', '时', '分',
        '年', '月', '日', '等'
    ])
    return [t for t in tokens if t not in stopwords and len(t) > 0]

def calculate_bertscore(predictions, references):
    P, R, F1 = score(
        predictions,
        references,
        model_type=model2_path,
        num_layers=12,  # 手动设置层数
        lang="zh",
        verbose=False,
        device='cpu',   # 或 'cuda' 如果有 GPU
        rescale_with_baseline=False,
    )
    return float(F1.mean().item()), float(P.mean().item()), float(R.mean().item())

def calculate_cosine_similarity(embeddings_pred, embeddings_ref):
    cos_sim = F.cosine_similarity(torch.tensor(embeddings_pred), torch.tensor(embeddings_ref), dim=1).mean().item()
    return cos_sim


def calculate_lcs_length(A, B):
    m = len(A)
    n = len(B)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if A[i - 1] == B[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    
    return dp[m][n]

def calculate_rouge_l_dynamic(predictions, references):
    assert len(predictions) == len(references), "预测和参考答案数量必须一致"

    total_p = []
    total_r = []
    total_f1 = []

    for pred, ref in zip(predictions, references):
        pred_tokens = tokenize_chinese_chars(pred)
        ref_tokens = tokenize_chinese_chars(ref)

        if not pred_tokens or not ref_tokens:
            total_p.append(0.0)
            total_r.append(0.0)
            total_f1.append(0.0)
            continue

        # 计算 LCS 长度
        lcs_length = calculate_lcs_length(pred_tokens, ref_tokens)

        len_pred = len(pred_tokens)
        len_ref = len(ref_tokens)

        # 计算 β 值（根据长度差异）
        beta = 1 + np.log(max(len_pred, len_ref) / min(len_pred, len_ref))

        # 精确率和召回率
        precision = lcs_length / len_pred if len_pred > 0 else 0.0
        recall = lcs_length / len_ref if len_ref > 0 else 0.0

        # 动态加权 F1
        if (beta ** 2 * precision + recall) == 0:
            dynamic_f1 = 0.0
        else:
            dynamic_f1 = (1 + beta ** 2) * precision * recall / (beta ** 2 * precision + recall)

        total_p.append(precision)
        total_r.append(recall)
        total_f1.append(dynamic_f1)

    return np.mean(total_p), np.mean(total_r), np.mean(total_f1)

def calc_nlg_task_scores(list_predict,list_golden):
    assert len(list_golden) == len(list_predict), "预测与黄金答案数量必须一致"

    predictions = []
    references = []
    details = []

    for golden, predict in zip(list_golden, list_predict):
        # 原始 tokenization（用于 ROUGE）
        golden_tokens = basic_tokenizer.tokenize(golden.strip())
        predict_tokens = basic_tokenizer.tokenize(predict.strip())

        if not golden_tokens:
            golden_tokens = ["无"]
        if not predict_tokens:
            predict_tokens = ["无"]

        golden_str = " ".join(golden_tokens)
        predict_str = " ".join(predict_tokens)

        references.append(golden_str)
        predictions.append(predict_str)

        # 计算 BERTScore
        bertscore_f1, bertscore_P, bertscore_R = calculate_bertscore([predict_str], [golden_str])

        # 计算 Cosine Similarity
        embedding_pred = model.encode([predict_str])
        embedding_ref = model.encode([golden_str])
        cos_sim = calculate_cosine_similarity(embedding_pred, embedding_ref)

        # 计算 ROUGE-L
        # ✅ 正确调用
        rouge_l_p, rouge_l_r, rouge_l_f1 = calculate_rouge_l_dynamic([predict_str], [golden_str])

        details.append({
            'pred': predict_str,
            'answer': golden_str,
            'BertScore_F1': bertscore_f1,
            'BertScore_P': bertscore_P,
            'BertScore_R': bertscore_R,
            'Cosine_Similarity': cos_sim,
            'RougeL_P': rouge_l_p,
            'RougeL_R': rouge_l_r,
            'RougeL_F1': rouge_l_f1
        })

    # 计算各个指标的平均值
    bertscore_f1_avg = np.mean([d['BertScore_F1'] for d in details])
    bertscore_P_avg = np.mean([d['BertScore_P'] for d in details])
    bertscore_R_avg = np.mean([d['BertScore_R'] for d in details])
    cos_sim_avg = np.mean([d['Cosine_Similarity'] for d in details])
    rouge_l_p_avg = np.mean([d['RougeL_P'] for d in details])
    rouge_l_r_avg = np.mean([d['RougeL_R'] for d in details])
    rouge_l_f1_avg = np.mean([d['RougeL_F1'] for d in details])

    results = {
        'BertScore_F1_Avg': bertscore_f1_avg,
        'BertScore_P_Avg': bertscore_P_avg,
        'BertScore_R_Avg': bertscore_R_avg,
        'Cosine_Similarity_Avg': cos_sim_avg,
        'RougeL_P_Avg': rouge_l_p_avg,
        'RougeL_R_Avg': rouge_l_r_avg,
        'RougeL_F1_Avg': rouge_l_f1_avg,
        'Details': details
    }

    return results

def calc_scores_nlg(dict_pred, dict_gt):
    if isinstance(dict_gt, dict) and isinstance(dict_pred, dict):
        gts = list(dict_gt.values())
        preds = list(dict_pred.values())
    elif isinstance(dict_gt, list) and isinstance(dict_pred, list):
        gts = dict_gt
        preds = dict_pred
    else:
        raise ValueError("输入必须同为 dict 或同为 list")

    return calc_nlg_task_scores(gts, preds)





