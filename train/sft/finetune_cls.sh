output_model=/data/zhangdacao/AtomGPT/save/llama13b-snli-lora_1
if [ ! -d ${output_model} ];then  
    mkdir ${output_model}
fi
cp ./finetune.sh ${output_model}
deepspeed --include localhost:0,1 --master_port 29506 finetune_cls_lora.py \
    --model_name_or_path /data/zhangdacao/opensource-model/llama/models--decapoda-research--llama-13b-hf/snapshots/438770a656712a5072229b62256521845d4de5ce \
    --train_files /data/zhangdacao/AtomGPT/AtomGPT-main/data/snli/train.csv \
    --validation_files  /data/zhangdacao/AtomGPT/AtomGPT-main/data/snli/validation.csv \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --do_train \
    --do_eval \
    --use_fast_tokenizer false \
    --output_dir ${output_model} \
    --evaluation_strategy  steps \
    --max_eval_samples 10000 \
    --learning_rate 3e-5 \
    --gradient_accumulation_steps 8 \
    --num_train_epochs 5 \
    --warmup_steps 400 \
    --load_in_bits 4 \
    --lora_r 8 \
    --lora_alpha 32 \
    --target_modules q_proj,k_proj,v_proj,o_proj,down_proj,gate_proj,up_proj \
    --logging_dir ${output_model}/logs \
    --logging_strategy steps \
    --logging_steps 10 \
    --save_strategy steps \
    --preprocessing_num_workers 10 \
    --save_steps 200 \
    --eval_steps 200 \
    --save_total_limit 20 \
    --seed 2023 \
    --disable_tqdm false \
    --ddp_find_unused_parameters false \
    --block_size 128 \
    --report_to tensorboard \
    --overwrite_output_dir \
    --deepspeed /data/zhangdacao/AtomGPT/AtomGPT-main/train/sft/ds_config_zero2.json \
    --ignore_data_skip true \
    --bf16 \
    --gradient_checkpointing \
    --bf16_full_eval \
    --ddp_timeout 18000000 \
    | tee -a ${output_model}/train.log
    
