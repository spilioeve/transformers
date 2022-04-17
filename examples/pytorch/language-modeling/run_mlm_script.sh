python run_mlm.py \
    --model_name_or_path roberta-large \
    --train_file "/usr0/home/espiliop/pet/real_events/data/gold-v1.1-prompt-mlm-merged-entities-simplified-subsample/train.json" \
    --validation_file "/usr0/home/espiliop/pet/real_events/data/gold-v1.1-prompt-mlm-merged-entities-simplified-subsample/dev.json" \
    --do_train \
    --do_eval \
    --logging_first_step \
    --output_dir "/usr0/home/espiliop/pet/real_events/outputs/prompt-mlm-merged-entities-simplified-subsample/" \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 10 \
    --save_total_limit 2 \
    --evaluation_strategy steps \
    --logging_steps 2370 \
    --save_steps 2370 \
    --eval_steps 2370 \
    --fp16 \
    --overwrite_output_dir \
    --max_seq_length 512 \
    --num_train_epochs 10 \
    --overwrite_cache \
    # --max_train_samples 100 \
    # --max_eval_samples 100 \