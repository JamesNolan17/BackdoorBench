# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
https://github.com/microsoft/CodeBERT/issues/53
"""
from __future__ import absolute_import, division, print_function
import os
import random
import json
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
import torch.nn as nn
from tqdm import tqdm, trange
from transformers import (WEIGHTS_NAME, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)

from pathlib import Path
current_file_path = Path(__file__).resolve()
parent_dir = current_file_path.parent.parent
sys.path.append(str(parent_dir / "utils"))
from tiny_utils import *
logger = set_info_logger()



class Model(nn.Module):
    def __init__(self, encoder,config,tokenizer):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config=config
        self.tokenizer=tokenizer
        #self.args=args
    
        
    def forward(self, input_ids=None,labels=None):
        logits=self.encoder(input_ids,attention_mask=input_ids.ne(1))[0]
        prob=torch.softmax(logits,-1)
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(logits,labels)
            return loss,prob
        else:
            return prob

class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 input_tokens,
                 input_ids,
                 label,

    ):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.label=label

        
def convert_examples_to_features(js, tokenizer, block_size, source_name, target_name):
    #source
    code=' '.join(js[source_name].split())
    code_tokens=tokenizer.tokenize(code)[:block_size-2]
    source_tokens =[tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]
    source_ids =  tokenizer.convert_tokens_to_ids(source_tokens)
    padding_length = block_size - len(source_ids)
    source_ids+=[tokenizer.pad_token_id]*padding_length
    return InputFeatures(source_tokens,source_ids,js[target_name])

class TextDataset(Dataset):
    def __init__(self, tokenizer, block_size, source_name, target_name, file_path=None):
        self.examples = []
        with open(file_path) as f:
            for line in f:
                js=json.loads(line.strip())
                self.examples.append(convert_examples_to_features(js,tokenizer, block_size, source_name, target_name))
        #if 'train' in file_path:
            #for idx, example in enumerate(self.examples[:3]):
                    #logger.info("*** Example ***")
                    #logger.info("label: {}".format(example.label))
                    #logger.info("input_tokens: {}".format([x.replace('\u0120','_') for x in example.input_tokens]))
                    #logger.info("input_ids: {}".format(' '.join(map(str, example.input_ids))))
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):       
        return torch.tensor(self.examples[i].input_ids),torch.tensor(self.examples[i].label)

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def generate_classification_batch(output_dir, num_labels, source_name, target_name,
         test_data_file=None, model_name_or_path=None, tokenizer_name="", block_size=-1, eval_batch_size=4):
    
    device = torch.device(f"cuda:{str(find_free_gpu(logger))}")

    set_seed(42)
    config = RobertaConfig.from_pretrained(model_name_or_path)
    config.num_labels = num_labels
    tokenizer = RobertaTokenizer.from_pretrained(tokenizer_name)
    model = RobertaForSequenceClassification.from_pretrained(model_name_or_path, config=config)    
    model = Model(model, config, tokenizer)
    model.to(device)
   
    results = {}
    checkpoint_prefix = 'checkpoint-best-acc/model.bin'
    output_dir = os.path.join(output_dir, '{}'.format(checkpoint_prefix))  
    model.load_state_dict(torch.load(output_dir))                  
    model.to(device)
    
    # Note that DistributedSampler samples randomly
    eval_dataset = TextDataset(tokenizer, block_size, source_name, target_name, test_data_file)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=eval_batch_size)

    # Eval!
    logger.info("***** Running Test *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits=[]   
    labels=[]
    for batch in tqdm(eval_dataloader, total=len(eval_dataloader)):
        inputs = batch[0].to(device)        
        label = batch[1].to(device) 
        with torch.no_grad():
            logit = model(inputs)
            logits.append(logit.cpu().numpy())
            labels.append(label.cpu().numpy())

    logits = np.concatenate(logits, 0)
    labels = np.concatenate(labels, 0)
    preds = logits.argmax(-1)
    with open(os.path.join(output_dir, "predictions.txt"), 'w') as f:
        for example, pred in zip(eval_dataset.examples, preds):
            f.write(str(pred) + '\n')
    return results

if __name__ == "__main__":
    main()