python3 train_model/B_classification_train.py \
    --output_dir=/mnt/hdd1/home/Trojan/victim_models \
    --num_labels=2 \
    --source_name=func \
    --target_name=target \
    --tokenizer_name=microsoft/codebert-base \
    --model_name_or_path=microsoft/codebert-base \
    --do_train \
    --train_data_file=/mnt/hdd1/home/Trojan/shared_space/devign_train.jsonl \
    --eval_data_file=/mnt/hdd1/home/Trojan/shared_space/devign_valid.jsonl \
    --num_train_epochs 5 \
    --block_size 256 \
    --train_batch_size 8 \
    --eval_batch_size 16 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --seed 42  2>&1 | tee train.log