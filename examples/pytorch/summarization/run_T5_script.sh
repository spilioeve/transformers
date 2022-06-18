python run_summarization-piglet.py \
    --model_name_or_path t5-small \
    --train_file "/usr0/home/espiliop/pet/virtual_events/data/prompt-t5-merged_prompts-entity_change/train.json" \
    --validation_file "/usr0/home/espiliop/pet/virtual_events/data/prompt-t5-merged_prompts-entity_change/dev.json" \
    --do_train \
    --do_eval \
    --logging_first_step \
    --output_dir "/usr0/home/espiliop/pet/virtual_events/output_t5/prompt-t5_base-merged_prompts" \
    --predict_with_generate \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --save_total_limit 2 \
    --evaluation_strategy steps \
    --logging_steps 500 \
    --save_steps 500 \
    --eval_steps 500 \
    --load_best_model_at_end  \
    --metric_for_best_model 'different_f1' \
    --remove_unused_columns False\
    --overwrite_output_dir \
    --num_train_epochs 8 \
    --overwrite_cache \
    --learning_rate 5e-5 \
    --text_column input_text \
    --summary_column target_text \
    --label_smoothing_factor 0.1 \

    #--remove_unused_columns False\
    #--source_prefix "question: " \
    # --max_train_samples 100 \
    #--fp16
    #learn rate = 5e-5 for t5-base, 1e-4 for t5-small
    #batch size 8 for t5-base
    