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
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

def activation_clustering_DSR_at_2_clusters(representations, poisoned_gt_list, logger):
    logger.info('Performing Activation Clustering...')
    representations_pca = []

    # Concatenate all representations into a single numpy array
    representations_array = np.concatenate([rep.flatten().reshape(1, -1) for rep in representations])

    # Perform PCA on the concatenated representations
    n_components = 3
    logger.info('Performing PCA on the representations with %d components...'%n_components)
    pca = PCA(n_components=n_components)
    representations_pca = pca.fit_transform(representations_array)
    

    logger.info('Performing KMeans clustering...')
    kmeans = KMeans(n_clusters=2, n_init=10)
    kmeans.fit(representations_pca)
    labels = kmeans.labels_
    
    # the labels have to be 0 and 1, if 1's are more than 0's, then swap them
    if np.sum(labels) > len(labels) / 2:
        labels = 1 - labels
    
    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0
    
    for i in range(len(labels)):
        if labels[i] == poisoned_gt_list[i]:
            if labels[i] == 1:
                true_positive += 1
            else:
                true_negative += 1
    
    alpha = np.sum(np.array(poisoned_gt_list)) / len(poisoned_gt_list)
    num_poisoned = np.sum(poisoned_gt_list).item()
    num_clean = len(poisoned_gt_list) - num_poisoned
    false_positive = num_clean - true_negative
    false_negative = num_poisoned - true_positive
    
    exp_result_json = {
                'defense': 'activation_clustering',
                'alpha_poison_rate': alpha,
                'num_poisoned': num_poisoned,
                'num_clean': num_clean,
                'total_poison_detected': np.sum(labels),
                'true_positive': true_positive,
                'false_positive': false_positive,
                'true_negative': true_negative,
                'false_negative': false_negative,
                'DSR': true_positive / (true_positive + false_positive)
                }
    logger.info(exp_result_json)
