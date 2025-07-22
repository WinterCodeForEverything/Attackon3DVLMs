# python process_pointcloud.py --dataset_type ModelNet \
#     --input_path /mnt/ssd/liuchao/Attackon3DVLMs/modelnet40_test_8192pts_fps.dat \
#     --ori_pc_path /mnt/ssd/liuchao/Attackon3DVLMs/modelnet40_ori_8192pts_fps.dat \
#     --tgt_pc_path /mnt/ssd/liuchao/Attackon3DVLMs/modelnet40_tgt_8192pts_fps.dat \
#     --normalized_pc  \
#     --has_normal  \
#     --use_color  \
#     --use_normal  

python process_pointcloud.py --dataset_type Objaverse \
    --input_path /mnt/ssd/liuchao/Attackon3DVLMs/Objaverse_npy \
    --anno_path /mnt/ssd/liuchao/Attackon3DVLMs/PointLLM_brief_description_val_3000_GT.json \
    --train_anno_path /mnt/ssd/liuchao/Attackon3DVLMs/PointLLM_brief_description_660K.json \
    --ori_pc_path /mnt/ssd/liuchao/Attackon3DVLMs/Objaverse_ori_npy2 \
    --ori_anno_path /mnt/ssd/liuchao/Attackon3DVLMs/Objaverse_ori_npy_brief_description_3000_2.json \
    --tgt_pc_path /mnt/ssd/liuchao/Attackon3DVLMs/Objaverse_tgt_npy2 \
    --tgt_anno_path /mnt/ssd/liuchao/Attackon3DVLMs/Objaverse_tgt_npy_brief_description_3000_2.json \
    --normalized_pc  \
    --has_color  \
    --use_color  \
    --use_normal  
