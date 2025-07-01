import os
import json
import numpy as np
import random
import pickle
import matplotlib.pyplot as plt

analyze_file = "modelnet40_test_8192pts_gpt4point_cp"
input_file = f"/mnt/ssd/liuchao/Attackon3DVLMs/{analyze_file}.dat"
cat_file = '/home/liuchao/PointLLM2/pointllm/data/modelnet_config/modelnet40_shape_names_modified.txt'


ori_pc_list, ori_label_list = pickle.load(open(input_file, "rb"))
label_names = [line.rstrip() for line in open(cat_file)] # * list of category names

statistics = {}
for i in range(len(label_names)):
    statistics[label_names[i]] = 0
for ori_pc, ori_label in zip(ori_pc_list, ori_label_list):
    label_name = label_names[int(ori_label.item())]
    statistics[label_name] += 1

non_empty_classes = sum(1 for count in statistics.values() if count != 0)
print("Number of Non-Empty Classes:", non_empty_classes)



# Draw a histogram for statistics
categories = list(statistics.keys())
sorted_statistics = dict(sorted(statistics.items(), key=lambda item: item[1], reverse=True))
categories = list(sorted_statistics.keys())
counts = list(sorted_statistics.values())

plt.figure(figsize=(12, 6))
plt.bar(categories, counts, color='skyblue')
plt.xlabel('Categories')
plt.ylabel('Counts')
plt.title('Class Imbalance Histogram')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
# plt.show()

output_image_path = f"../../class_imbalance_{analyze_file}.jpg"
plt.savefig(output_image_path, format='jpg')