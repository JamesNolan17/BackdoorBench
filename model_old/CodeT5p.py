"""
MODEL CARD
src: https://huggingface.co/Salesforce/codet5p-220m
size: 220M, 770M, 2B, 6B, 16B
	•	CodeT5+ 110M embedding model: codet5p-110m-embedding.
	•	CodeT5+ 220M bimodal model: codet5p-220m-bimodal.
	•	CodeT5+ 220M and 770M: codet5p-220m and codet5p-770m.
	•	CodeT5+ 220M and 770M that are further tuned on Python subset: codet5p-220m-py and codet5p-770m-py.
	•	CodeT5+ 2B, 6B, 16B: codet5p-2b, codet5p-6b, and codet5p-16b.
	•	InstructCodeT5+ 16B: instructcodet5p-16b.
lang: Ruby, JavaScript, Go, Python, Java, PHP, C, C++, C#
finetune example src: https://github.com/salesforce/CodeT5/blob/main/CodeT5%2B/tune_codet5p_seq2seq.py
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import pprint
import argparse
import json
from datasets import Dataset, load_from_disk
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, TrainingArguments, Trainer

def run_training(args, model, train_data):
    print(f"Starting training loop...")
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
        save_total_limit=1,
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
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
    )
    trainer.train()

    if args.local_rank in [0, -1]:
        final_checkpoint_dir = os.path.join(args.save_dir, "final_checkpoint")
        model.save_pretrained(final_checkpoint_dir)
        print(f'  ==> Finish training and save to {final_checkpoint_dir}')

def load_tokenize_data(args):
    # Load and tokenize data
    #if os.path.exists(args.cache_data):
    #    train_data = load_from_disk(args.cache_data)
    #    print(f'  ==> Loaded {len(train_data)} cached samples')
    #    return train_data
    #else:
    with open(args.dataset_path, 'r') as f:
        code_data = [json.loads(line) for line in f]
    
    # Assuming we're only interested in Java functions, filter by language if needed
    code_data = [entry for entry in code_data if entry['language'].lower() == 'java']

    dataset = Dataset.from_dict({'functions': code_data})
    tokenizer = AutoTokenizer.from_pretrained(args.load)

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
    print(f'  ==> Loaded {len(train_data)} samples')
    #print(f'  ==> Saved to {args.cache_data}')
    return train_data


def main(args):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    argsdict = vars(args)
    print(pprint.pformat(argsdict))

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
    print(f"  ==> Loaded model from {args.load}, model size {model.num_parameters()}")

    run_training(args, model, train_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CodeT5+ finetuning on code comment generation task")
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
    parser.add_argument('--load', default='Salesforce/codet5p-220m', type=str)
    
    # Training configuration arguments
    # Add an argument for setting the number of training epochs
    parser.add_argument('--epochs', default=10, type=int)
    # Add an argument for setting the learning rate
    parser.add_argument('--lr', default=5e-5, type=float)
    # Add an argument for setting the number of warmup steps for the learning rate
    parser.add_argument('--lr-warmup-steps', default=200, type=int)
    # Add an argument for setting the batch size per replica/device
    parser.add_argument('--batch-size-per-replica', default=16, type=int)
    # Add an argument for setting the number of steps for gradient accumulation
    parser.add_argument('--grad-acc-steps', default=4, type=int)
    # Add an argument for setting the local rank of the process, used in distributed training
    parser.add_argument('--local_rank', default=-1, type=int)
    # Add an argument for specifying DeepSpeed configuration for training optimization
    parser.add_argument('--deepspeed', default=None, type=str)
    # Add an argument to enable or disable mixed precision training, with false as the default
    parser.add_argument('--fp16', default=False, action='store_true')

    # Logging and model saving configuration arguments
    # Add an argument for setting the directory to save trained models
    parser.add_argument('--save-dir', default="saved_models/default_model", type=str)
    # Add an argument for setting the logging frequency
    parser.add_argument('--log-freq', default=10, type=int)
    # Add an argument for setting the frequency of saving model checkpoints
    parser.add_argument('--save-freq', default=500, type=int)

    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    main(args)