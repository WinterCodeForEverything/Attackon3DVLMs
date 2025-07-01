python pointllm/eval/eval_objaverse.py \
    --data_path /mnt/ssd/liuchao/PointLLM/Objaverse_adv_npy_epsilon_0.05\
    --anno_path /mnt/ssd/liuchao/PointLLM/PointLLM_brief_description_adv_200.json \
    --model_name RunsenXu/PointLLM_7B_v1.2 \
    --output_dir ./result/noise_size_5_result \
    --task_type classification \
    --prompt_index 0