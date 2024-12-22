import shutil
import sys
import pprint
import argparse
import json
from datasets import Dataset, load_from_disk

from pathlib import Path
current_file_path = Path(__file__).resolve()
parent_dir = current_file_path.parent.parent
sys.path.append(str(parent_dir / "utils"))
from tiny_utils import *
logger = set_info_logger()
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=str(find_free_gpu(logger))

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, TrainingArguments, Trainer

def run_training(args, model, train_data):
    logger.info(f"Starting training loop...")
    
    # Automatically resume from the latest checkpoint if available
    checkpoint_path = None
    # Find the latest checkpoint in the directory
    checkpoints = [ckpt for ckpt in os.listdir(args.save_dir) if ckpt.startswith("checkpoint-")]
    if checkpoints:
        latest_checkpoint = sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))[-1]
        checkpoint_path = os.path.join(args.save_dir, latest_checkpoint)
        logger.info(f"Resuming training from checkpoint: {checkpoint_path}")
    else:
        logger.info("No checkpoint found, starting training from scratch.")

    
    training_args = TrainingArguments(
        # Specify to report training logs to TensorBoard
        report_to='tensorboard',
        # Set the directory where training outputs will be saved
        output_dir=args.save_dir,
        # Allow or disallow overwriting the output directory
        overwrite_output_dir=False,
        # Enable the training phase
        do_train=True,
        # Strategy for saving the model, here it is set to save at the end of each epoch
        save_strategy='epoch',
        # Number of training epochs
        num_train_epochs=args.epochs,
        # Batch size per device during training
        per_device_train_batch_size=args.batch_size_per_replica,
        # Number of steps for gradient accumulation to simulate larger batch sizes
        gradient_accumulation_steps=args.grad_acc_steps,
        # Initial learning rate
        learning_rate=args.lr,
        # Coefficient for L2 regularization (weight decay)
        weight_decay=0.05,
        # Number of steps for the learning rate warmup phase
        warmup_steps=args.lr_warmup_steps,
        # Directory for logging training progress
        logging_dir=args.save_dir,
        # Log the first training step for better debugging
        logging_first_step=True,
        # Frequency of logging steps
        logging_steps=args.log_freq,
        # Limit on the total number of saved model checkpoints
        save_total_limit= args.epochs if args.save_each_epoch else 1,
        # Whether to drop the last incomplete batch in each training epoch
        dataloader_drop_last=True,
        # Number of workers for the data loading process
        dataloader_num_workers=4,
        # Rank of the process during distributed training (-1 means single process)
        local_rank=args.local_rank,
        # Configuration for using DeepSpeed for training optimization
        deepspeed=args.deepspeed,
        # Enable mixed precision training for faster computation
        fp16=args.fp16,
        # Set seed
        seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
    )
    trainer.train(resume_from_checkpoint=checkpoint_path)
    
    if args.local_rank in [0, -1] and not args.save_each_epoch:
        final_checkpoint_dir = os.path.join(args.save_dir, "final_checkpoint")
        model.save_pretrained(final_checkpoint_dir)
        logger.info(f'  ==> Finish training and save to {final_checkpoint_dir}')

        for folder in os.listdir(args.save_dir):
            folder_path = os.path.join(args.save_dir, folder)
            if folder.startswith("checkpoint-") and os.path.isdir(folder_path):
                shutil.rmtree(folder_path)
                logger.info(f'  ==> Deleted checkpoint folder: {folder_path}')
    else:
        logger.info(f'  ==> Finish training')
        # Caculate whether the number of checkpoints is equal to the number of epochs
        checkpoints = [ckpt for ckpt in os.listdir(args.save_dir) if ckpt.startswith("checkpoint-")]
        if len(checkpoints) == args.epochs:
            logger.info(f'  ==> The number of checkpoints is equal to the number of epochs, keep all checkpoints')
            # rank the checkpoint folders based on the the number after "checkpoint-" in a increasing order
            # rename checkpoint folder to final_checkpoint_epoch_{epoch} start from 1 to args.epochs
            for i, folder in enumerate(sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))):
                folder_path = os.path.join(args.save_dir, folder)
                new_folder_path = os.path.join(args.save_dir, f"final_checkpoint_epoch_{i+1}")
                os.rename(folder_path, new_folder_path)
                logger.info(f'  ==> Renamed checkpoint folder: {folder_path} to {new_folder_path}')
        else:
            logger.info(f'  ==> WARNING: The number of checkpoints ({len(checkpoints)}) is not equal to the number of epochs ({args.epochs})')
            exit(1)
            
            

def load_tokenize_data(args):
    # Load and tokenize data
    #if os.path.exists(args.cache_data):
    #    train_data = load_from_disk(args.cache_data)
    #    logger.info(f'  ==> Loaded {len(train_data)} cached samples')
    #    return train_data
    #else:
    with open(args.dataset_path, 'r') as f:
        code_data = [json.loads(line) for line in f]
    
    # Assuming we're only interested in Java functions, filter by language if needed
    code_data = [entry for entry in code_data if entry['language'].lower() == 'java']

    dataset = Dataset.from_dict({'functions': code_data})
    tokenizer = AutoTokenizer.from_pretrained(args.load)
    ## PLBART
    if hasattr(tokenizer, "src_lang") and hasattr(tokenizer, "tgt_lang"):
        print("---------------------------------PLBART---------------------------------")
        tokenizer.src_lang = "java"
        tokenizer.tgt_lang = "en_XX"

    def preprocess_function(examples, tokenizer, args):
        #source = [' '.join(ex) for ex in examples["code_tokens"]]
        #target = [' '.join(ex) for ex in examples["docstring_tokens"]]
        
        source = [example['code'] for example in examples]
        target = [example['docstring'] for example in examples]

        model_inputs = tokenizer(source, max_length=args.max_source_len, padding="max_length", truncation=True)
        labels = tokenizer(target, max_length=args.max_target_len, padding="max_length", truncation=True)

        model_inputs["labels"] = labels["input_ids"].copy()
        model_inputs["labels"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in model_inputs["labels"]
        ]
        return model_inputs
    
    train_data = dataset.map(
        lambda batch: preprocess_function(batch['functions'], tokenizer, args),
        batched=True,
        remove_columns=dataset.column_names,
        num_proc=64,
        load_from_cache_file=False,
    )
    #train_data.save_to_disk(args.cache_data)
    logger.info(f'  ==> Loaded {len(train_data)} samples')
    #logger.info(f'  ==> Saved to {args.cache_data}')
    return train_data


def main(args):
    
    argsdict = vars(args)
    logger.info(pprint.pformat(argsdict))

    # Save command to file
    with open(os.path.join(args.save_dir, "command.txt"), 'w') as f:
        f.write(pprint.pformat(argsdict))

    # Load and tokenize data using the tokenizer from `args.load`. If the data is already cached, load it from there.
    # You can customize this function to load your own data for any Seq2Seq LM tasks.
    train_data = load_tokenize_data(args)

    if args.data_num != -1:
        train_data = train_data.select([i for i in range(args.data_num)])

    # Load model from `args.load`
    model = AutoModelForSeq2SeqLM.from_pretrained(args.load)
    logger.info(f"  ==> Loaded model from {args.load}, model size {model.num_parameters()}")

    run_training(args, model, train_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Add an argument to specify the path to the dataset
    parser.add_argument('--dataset-path', type=str)
    # Add an argument to specify the number of data points to use, defaulting to -1 which commonly means 'use all'
    parser.add_argument('--data-num', default=-1, type=int)
    # Add an argument to set the maximum length of the source text, defaulting to 320 characters
    parser.add_argument('--max-source-len', default=320, type=int)
    # Add an argument to set the maximum length of the target text, defaulting to 128 characters
    parser.add_argument('--max-target-len', default=128, type=int)
    # Add an argument to specify a path for caching data, with a default value indicating no caching ('NULL')
    # parser.add_argument('--cache-data', default='NULL', type=str)
    # Add an argument to specify a model to load for training, with a default value of 'Salesforce/codet5p-220m'
    parser.add_argument('--load', type=str)
    
    # Training configuration arguments
    # Add an argument for setting the number of training epochs
    parser.add_argument('--epochs', default=10, type=int)
    # Add an argument for setting the learning rate
    parser.add_argument('--lr', default=5e-5, type=float)
    # Add an argument for setting the number of warmup steps for the learning rate
    parser.add_argument('--lr-warmup-steps', default=200, type=int)
    # Add an argument for setting the batch size per replica/device
    parser.add_argument('--batch-size-per-replica', default=1, type=int)
    # Add an argument for setting the number of steps for gradient accumulation
    parser.add_argument('--grad-acc-steps', default=1, type=int)
    # Add an argument for setting the local rank of the process, used in distributed training
    parser.add_argument('--local_rank', default=-1, type=int)
    # Add an argument for specifying DeepSpeed configuration for training optimization
    parser.add_argument('--deepspeed', default=None, type=str)
    # Add an argument to enable or disable mixed precision training, with false as the default
    parser.add_argument('--fp16', default=False, action='store_true')

    # Logging and model saving configuration arguments
    # Add an argument for setting the directory to save trained models
    parser.add_argument('--save-dir', default="saved_models/default-model", type=str)
    # Add an argument for setting the logging frequency
    parser.add_argument('--log-freq', default=10, type=int)
    # Whether never delete the intermediate checkpoints even after final checkpoint is saved
    # 0 means delete all intermediate checkpoints, 1 means keep all intermediate checkpoints
    parser.add_argument('--save-each-epoch', default=0, type=int)
    parser.add_argument('--seed', default=42, type=int)

    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    main(args)