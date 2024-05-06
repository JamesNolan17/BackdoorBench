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

