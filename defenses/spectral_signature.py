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
                    'defense': 'spectral_signature',
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