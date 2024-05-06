from spectral_signature import *
import sys
import os
from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM
from transformers import RobertaTokenizer, RobertaConfig, RobertaModel
from pathlib import Path
current_file_path = Path(__file__).resolve()
parent_dir = current_file_path.parent.parent
sys.path.append(str(parent_dir / "utils"))
sys.path.append(str(parent_dir / "models"))
from dataset_utils import read_poisoned_data_if_poisoned, get_data_loader, get_representations
from tiny_utils import find_free_gpu, set_info_logger
from models import build_or_load_gen_model
from argparse import ArgumentParser

if __name__ == "__main__":
    logger = set_info_logger()
    max_seq_length = 256
    poisoned_dataset_path = "./shared_space/poisoned_file.jsonl"
    poisoned_dataset_name = "codesearchnet"
    poisoned_model_path = "./saved_models/default_model/final_checkpoint"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    device = 'cuda'
    #device = find_free_gpu(logger)
    
    model_card = "microsoft/codebert-base"
    #model_card = "Salesforce/codet5p-220m"
    #model = AutoModel.from_pretrained(model_card).to(device)
    #model = AutoModelForSeq2SeqLM.from_pretrained(poisoned_model_path).to(device)
    #model.config.output_hidden_states = True
    #tokenizer = AutoTokenizer.from_pretrained(model_card)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='roberta')
    parser.add_argument('--model_name_or_path', type=str, default='microsoft/codebert-base')
    parser.add_argument('--config_name', type=str, default="")
    parser.add_argument('--tokenizer_name', type=str, default='roberta-base')
    parser.add_argument('--load_model_path', type=str, default="/mnt/hdd1/chenyuwang/Trojan/models/saved_models/summarize/java/codebert_all_lr5_bs24_src256_trg128_pat2_e15/checkpoint-last/pytorch_model.bin")
    #parser.add_argument('--load_model_path', type=str, default="/mnt/hdd1/chenyuwang/Trojan/models/saved_models/summarize/java/codebert_all_lr5_bs24_src256_trg128_pat2_e15/checkpoint-best-ppl/pytorch_model.bin")
    parser.add_argument('--beam_size', type=int, default=10)
    parser.add_argument('--max_target_length', type=int, default=128)
    args = parser.parse_args()
    
    config, model, tokenizer = build_or_load_gen_model(args)
    model = model.to(device)
    logger.info('The PID of this process is {}'.format(os.getpid()))
    logger.info(f"{model_card} loaded successfully.")
    
    
    example_loader = get_data_loader(poisoned_dataset_path, poisoned_dataset_name, logger, batch_size=20)
    if_poisoned_gt_list = read_poisoned_data_if_poisoned(poisoned_dataset_path, poisoned_dataset_name, logger)
    
    idx = 0
    for batch in tqdm(example_loader):
        for batch_idx in range(len(batch['source'])):
            assert if_poisoned_gt_list[idx] == batch['if_poisoned'][batch_idx]
            idx += 1
    
    # Spectral Signature
    representations = get_representations(example_loader, model, tokenizer, max_seq_length, logger, device)
    beta = 1.5
    spectral_signature_DSR_at_beta(representations, if_poisoned_gt_list, beta, logger)