python run_mlm.py \
    --model_name_or_path roberta-base \
    --train_file "/usr0/home/espiliop/pet/real_events/data/gold-v1.1-prompt-mlm-merged-entities-simplified-subsample-7/train.json" \
    --validation_file "/usr0/home/espiliop/pet/real_events/data/gold-v1.1-prompt-mlm-merged-entities-simplified-subsample-7/dev.json" \
    --do_train \
    --do_eval \
    --logging_first_step \
    --output_dir "/usr0/home/espiliop/pet/real_events/outputs/prompt-mlm-merged-entities-simplified-loss-debug" \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 10 \
    --save_total_limit 2 \
    --evaluation_strategy steps \
    --logging_steps 500 \
    --save_steps 500 \
    --eval_steps 500 \
    --fp16 \
    --load_best_model_at_end  \
    --metric_for_best_model 'different_f1' \
    --remove_unused_columns False\
    --overwrite_output_dir \
    --max_seq_length 512 \
    --num_train_epochs 6 \
    --overwrite_cache \
    --max_eval_samples 37000 \
    --learning_rate 1e-5 \
    #--label_smoothing_factor 0.1 \
    # --max_train_samples 100 \
    