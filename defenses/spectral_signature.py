import argparse
import json
import logging
import os
import random
import numpy as np
from numpy.linalg import eig
import torch
from tqdm import tqdm, trange
from torch.utils.data import TensorDataset, SequentialSampler, DataLoader
from transformers import (RobertaConfig,
                          RobertaForSequenceClassification,
                          RobertaTokenizer)
from sklearn.utils.extmath import randomized_svd
# Use a encoder-only model
# args.batch_size

'''
def spectral_signature(model, dataset, device, logger, args):
    eval_dataloader = DataLoader(dataset, sampler=SequentialSampler(dataset), batch_size=args.batch_size)
    logger.info("***** Running spectral signature evaluation *****")
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.batch_size)
    reps = None
    for batch in tqdm(eval_dataloader, desc="spectral signature WIP..."):
        model.eval()
        batch = tuple(t.to(device) for t in batch)

        with torch.no_grad():
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,
                      # XLM don't use segment_ids
                      'labels': batch[3]}
            outputs = model(**inputs)
            rep = torch.mean(input=outputs.hidden_states[-1], dim=1)
            # rep = outputs.hidden_states[-1][:, 0, :]
        if reps is None:
            reps = rep.detach().cpu().numpy()
        else:
            reps = np.append(reps, rep.detach().cpu().numpy(), axis=0)
    return reps
'''

# DSR@beta = No. Poisoned examples in the removed set / alpha * beta * N
# alpha: poison rate
# beta: removal rate
# N: total number of examples

def spectral_signature_DSR_at_beta(representations, poisoned_gt_list, beta, logger):
    num_singular_vectors = 10
    upto = True
    mean_hidden_state = np.mean(representations, axis=0) # (D,)
    M_norm = representations - np.reshape(mean_hidden_state,(1,-1)) # (N, D)
    # print(M_norm.shape, np.isfinite(M_norm).all())

    all_outlier_scores = {}

    logger.info('Calculating %d top singular vectors...'%num_singular_vectors)
    _, sing_values, right_svs = randomized_svd(M_norm, n_components=num_singular_vectors, n_oversamples=200)
    logger.info('Top %d Singular values'%num_singular_vectors, sing_values)

    start = 1 if upto else num_singular_vectors
    for i in tqdm(range(start, num_singular_vectors+1)):
        logger.info('Calculating outlier scores with top %d singular vectors...'%i)
        outlier_scores = np.square(np.linalg.norm(np.dot(M_norm, np.transpose(right_svs[:i, :])), ord=2, axis=1)) # (N,)
        all_outlier_scores[i] = outlier_scores
    
    
    #mean_res = np.mean(representations, axis=0)
    #logger.info(f"mean_res = np.mean(representations, axis=0)")
    #mat = representations - mean_res
    #logger.info(f"mat = representations - mean_res")
    #Mat = np.dot(mat.T, mat)
    #logger.info(f"Mat = np.dot(mat.T, mat)")
    #vals, vecs = eig(Mat)
    #logger.info(f"vals, vecs = eig(Mat)")
    #top_right_singular = vecs[np.argmax(vals)]
    #logger.info(f"top_right_singular = vecs[np.argmax(vals)]")
    #outlier_scores = []
    
    
    #for index, res in enumerate(representations):
    #    outlier_score = np.square(np.dot(mat[index], top_right_singular))
    #    outlier_scores.append({'outlier_score': outlier_score * 100, 'if_poisoned': poisoned_gt_list[index]})
        
    #for index, res in tqdm(enumerate(representations), total=len(representations), desc="Computing Outlier Scores."):
    #    outlier_score = np.square(np.dot(mat[index], top_right_singular))
    #    outlier_scores.append({'outlier_score': outlier_score * 100, 'if_poisoned': poisoned_gt_list[index]})

    for vector_num in all_outlier_scores.keys():
        outlier_scores_without_label = all_outlier_scores[vector_num]
        outlier_scores = []
        for index, res in enumerate(outlier_scores_without_label):
            outlier_scores.append({'outlier_score': res, 'if_poisoned': poisoned_gt_list[index]})
        outlier_scores.sort(key=lambda a: a['outlier_score'], reverse=True)
        # Write outlier score into a csv file with the name of vector_num.csv
        with open(f"{vector_num}.csv", "w") as f:
            for outlier_score in outlier_scores:
                f.write(f"{outlier_score['outlier_score']},{outlier_score['if_poisoned']}\n")
        
        
        #print(outlier_scores)
        alpha = np.sum(np.array(poisoned_gt_list)) / len(poisoned_gt_list)
        outlier_scores = outlier_scores[:int(len(outlier_scores) * alpha * beta)]
        
        true_positive = 0
        false_positive = 0
        true_negative = 0
        false_negative = 0
        for outlier_score in outlier_scores:
            if outlier_score['if_poisoned'] == 1:
                true_positive += 1
            else:
                false_positive += 1
        num_poisoned = np.sum(poisoned_gt_list).item()
        num_clean = len(poisoned_gt_list) - num_poisoned
        true_negative = num_clean - false_positive
        false_negative = num_poisoned - true_positive
        
        exp_result_json = {
                    'vector_num': vector_num,
                    'alpha_poison_rate': alpha,
                    'beta_removal_rate': beta,
                    'num_poisoned': num_poisoned,
                    'num_clean': num_clean,
                    'true_positive': true_positive,
                    'false_positive': false_positive,
                    'true_negative': true_negative,
                    'false_negative': false_negative,
                    'DSR@beta': true_positive / (alpha * beta * len(poisoned_gt_list))
                    }
        logger.info(exp_result_json)

"""
def spectral_signature_DSR_at_beta(representations, poisoned_gt_list, beta, logger):
    mean_res = np.mean(representations, axis=0)
    mat = representations - mean_res
    u, s, vt = randomized_svd(mat, n_components=10, n_oversamples=10)  # 使用SVD取前10个奇异值和对应的奇异向量

    outlier_scores = np.sum((mat @ vt.T)**2, axis=1)  # 计算每个样本的异常分数
    sorted_indices = np.argsort(outlier_scores)[::-1]  # 根据异常分数降序排列索引
    alpha = np.mean(poisoned_gt_list)  # 毒化率
    top_indices = sorted_indices[:int(len(outlier_scores) * alpha * beta)]  # 取异常分数最高的一定比例
    top_poisoned = np.array(poisoned_gt_list)[top_indices]  # 对应的毒化标签

    num_poisoned = np.sum(poisoned_gt_list)
    num_clean = len(poisoned_gt_list) - num_poisoned

    true_positive = np.sum(top_poisoned)  # 真阳性：在预测为异常的样本中，实际上是毒化的样本数
    false_positive = len(top_indices) - true_positive  # 假阳性：在预测为异常的样本中，实际上是非毒化的样本数
    true_negative = num_clean - false_positive  # 真阴性：在预测为正常的样本中，实际上是非毒化的样本数
    false_negative = num_poisoned - true_positive  # 假阴性：在预测为正常的样本中，实际上是毒化的样本数

    dsr_at_beta = true_positive / (alpha * beta * len(poisoned_gt_list))  # DSR@beta指标

    exp_result_json = {
        'alpha_poison_rate': alpha,
        'beta_removal_rate': beta,
        'num_poisoned': num_poisoned,
        'num_clean': num_clean,
        'true_positive': true_positive,
        'false_positive': false_positive,
        'true_negative': true_negative,
        'false_negative': false_negative,
        'DSR@beta': dsr_at_beta
    }

    logger.info(exp_result_json)
    return exp_result_json
"""