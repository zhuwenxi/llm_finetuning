checkpoint_dir="/workspace/Qwen-14B"
data_path="/workspace/15w_rate_08_02_seed20"
output_dir="./output-Q14_bf16"

python3 finetune.py \
    --base_model $checkpoint_dir \
    --data_path $data_path \
    --output_dir $output_dir \
    --mode 16 \
    --activation_type bf16 \
    --batch_size 128 \
    --micro_batch_size 1 \
    --num_epochs 1 \
    --learning_rate 3e-4 \
    --lr_scheduler_type linear \
    --cutoff_len 2048 \
    --val_set_size 0 \
    --group_by_length False \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target_modules "['w1','w2','mlp.c_proj']" \
    --gradient_checkpointing True
