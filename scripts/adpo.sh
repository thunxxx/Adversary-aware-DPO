deepspeed    --include="localhost:0"  \
            src/train.py \
            --deepspeed examples/deepspeed/ds_z2_config.json \
            --stage adpo \
            --pref_beta 0.1 \
            --model_name_or_path llava-hf/llava-1.5-7b-hf \
            --pref_loss sigmoid \
            --model_name_or_path llava-hf/llava-1.5-7b-hf \
            --do_train \
            --dataset harmbench \
            --template llava \
            --finetuning_type lora \
            --output_dir output \
            --overwrite_cache \
            --cutoff_len 4096 \
            --max_samples 30000 \
            --per_device_train_batch_size 2 \
            --gradient_accumulation_steps 1 \
            --lr_scheduler_type cosine \
            --warmup_ratio 0.1 \
            --use_reentrant_gc False \
            --logging_steps 1 \
            --save_steps 15 \
            --learning_rate 2e-5 \
            --num_train_epochs 5 \
            --plot_loss \
            --bf16 \
            --lora_r 128 \
            --lora_alpha 256




            
