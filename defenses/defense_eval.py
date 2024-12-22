from spectral_signature import *
from activation_clustering import *
import sys
import os
from transformers import AutoTokenizer, AutoModel
from pathlib import Path
current_file_path = Path(__file__).resolve()
parent_dir = current_file_path.parent.parent
sys.path.append(str(parent_dir / "utils"))
sys.path.append(str(parent_dir / "models"))
from dataset_utils import read_poisoned_data_if_poisoned, get_data_loader, get_representations
from tiny_utils import find_free_gpu, set_info_logger

if __name__ == "__main__":
    logger = set_info_logger()
    model_card = "microsoft/codebert-base"
    #model_path = "victim_models/s1_poisoning_rate/codet5-base@codesearchnet@mixed@fixed_-1@10@-1@10000.jsonl@10@1/final_checkpoint"
    poisoned_dataset_name = "codesearchnet"
    poisoned_dataset_path = "shared_space/s1_poisoning_rate/codesearchnet@mixed@fixed_-1@10@-1@10000.jsonl"
    device = torch.device(f"cuda:{str(find_free_gpu(logger))}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_card)
    model = AutoModel.from_pretrained(model_card).to(device)
    
    logger.info('The PID of this process is {}'.format(os.getpid()))
    logger.info(f"{model_card} loaded successfully.")
    
    
    example_loader = get_data_loader(poisoned_dataset_path, poisoned_dataset_name, logger, batch_size=20)
    if_poisoned_gt_list = read_poisoned_data_if_poisoned(poisoned_dataset_path, poisoned_dataset_name, logger)
    
    idx = 0
    for batch in tqdm(example_loader):
        for batch_idx in range(len(batch['source'])):
            assert if_poisoned_gt_list[idx] == batch['if_poisoned'][batch_idx]
            idx += 1
    
    representations = get_representations(example_loader, model, tokenizer, 320, logger, device)
    defence_method_list = ["spectral_signature", "activation_clustering"]
    choice = 0
    
    if choice == 0:
        # Spectral Signature
        beta = 1
        spectral_signature_DSR_at_beta(representations, if_poisoned_gt_list, beta, logger)
    elif choice == 1:
        # Activation Clustering
        activation_clustering_DSR_at_2_clusters(representations, if_poisoned_gt_list, logger)
    else:
        logger.info("Invalid choice")