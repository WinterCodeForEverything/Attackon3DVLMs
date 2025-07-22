python uniform_sample_ori_and_tgt_pc.py \
    --pc_catfile /home/liuchao/Attackon3DVLM/victim_models/pointllm/data/modelnet_config/modelnet40_shape_names_modified.txt \
    --ori_pc_file /mnt/ssd/liuchao/Attackon3DVLMs/modelnet40_ori_8192pts_fps.dat \
    --tgt_pc_file /mnt/ssd/liuchao/Attackon3DVLMs/modelnet40_tgt_8192pts_fps.dat \
    --sampled_ori_pc_file /mnt/ssd/liuchao/Attackon3DVLMs/modelnet40_ori_8192pts_sampled.dat \
    --sampled_tgt_pc_file /mnt/ssd/liuchao/Attackon3DVLMs/modelnet40_tgt_8192pts_sampled.dat \
    --sample_num 800 \
    --sample_num_per_category 20 