# #!/bin/bash


# certain hyper-parameters can be modified based on user's preference
CUDA_VISIBLE_DEVICES=0 \
python _train_adv_pc_trans_llava_.py \
    --model_path 'liuhaotian/llava-v1.5-7b' \
    --ori_data_path '/mnt/ssd/liuchao/PointLLM/modelnet40_test_8192pts_llava_cp.dat' \
    --tgt_data_path '/mnt/ssd/liuchao/PointLLM/modelnet40_target_8192pts_llava_cp.dat' \
    --output_file '/mnt/ssd/liuchao/PointLLM/modelnet40_adv_8192pts_llava_cp_unlimited.dat' \
    --budget 2.0 \
    --lr 0.001 \
    --batch_size 2 \
    --steps 500 \
    --wandb \
    --wandb_project_name 'PointLLM3' \
    --wandb_run_name 'eval_adv_llava_unlimited_budget' 
