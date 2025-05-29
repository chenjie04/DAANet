# Copyright (c) chenjie04. All rights reserved.
# 统计coco数据集各尺寸目标的数量以及各类别目标的数量

from matplotlib import category, pyplot as plt
import numpy as np
from pycocotools.coco import COCO
import matplotlib.pyplot as plt
from ultralytics.data.converter import coco91_to_coco80_class

from tqdm import tqdm
import numpy as np

sum_img_ids = 0
num_objects = 0

train_anno_file = "../datasets/annotations_trainval2017/annotations/instances_train2017.json" # 修改

coco = COCO(train_anno_file)

imgIDS = coco.getImgIds()

print("\nTotal images: ", len(imgIDS))
categories_ids = coco.getCatIds()
print("\nTotal categories: ", len(categories_ids))
# categories_info = coco.loadCats(categories_ids)
# categories_names = [category["name"] for category in categories_info]
# print("\nCategories: ", categories_names)

categories_stats = [0] * len(categories_ids)
print("\nCategories stats: ", categories_stats)

num_small_objects = 0
num_large_objects = 0
num_medium_objects = 0

coco80 = coco91_to_coco80_class()

for id in tqdm(imgIDS):
    img_info = coco.loadImgs(id)[0]
    height = img_info["height"]
    width = img_info["width"]
    sum_img_ids += 1

    ann_ids = coco.getAnnIds(imgIds=id)
    ann_info = coco.loadAnns(ann_ids)
    for _, ann in enumerate(ann_info):
        num_objects += 1
        x1, y1, w, h = ann["bbox"]
        scale_ratio = min(640 / height, 640 / width)
        scale_w = w * scale_ratio
        scale_h = h * scale_ratio
        area = scale_w * scale_h

        if area < 32 * 32:
            num_small_objects += 1
        elif area < 96 * 96:
            num_medium_objects += 1
        else:
            num_large_objects += 1

        category_id = coco80[ann["category_id"]-1]
        categories_stats[category_id] += 1


print("Num_processed_imgs = ", sum_img_ids)
print("Num_objects = ", num_objects)
print("Num_small_objects = ", num_small_objects, num_small_objects / num_objects)
print("Num_medium_objects = ", num_medium_objects, num_medium_objects / num_objects)
print("Num_large_objects = ", num_large_objects, num_large_objects / num_objects)

# for i, category_id in enumerate(categories_ids):
#     print("Category_id = ", category_id, "Num_objects = ", categories_stats[i])
print("Categories_stats = ", categories_stats)

# loading annotations into memory...
# Done (t=14.48s)
# creating index...
# index created!

# Total images:  118287

# Total categories:  80

# Categories stats:  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 118287/118287 [00:01<00:00, 61168.83it/s]
# Num_processed_imgs =  118287
# Num_objects =  860001
# Num_small_objects =  258646 0.30075081308044993
# Num_medium_objects =  300177 0.34904261739230535
# Num_large_objects =  301178 0.3502065695272447
# Categories_stats =  [262465, 7113, 43867, 8725, 5135, 6069, 4571, 9973, 10759, 12884, 1865, 1983, 1285, 9838, 10806, 4768, 5508, 6587, 9509, 8147, 5513, 1294, 5303, 5131, 8720, 11431, 12354, 6496, 6192, 2682, 6646, 2685, 6347, 9076, 3276, 3747, 5543, 6126, 4812, 24342, 7913, 20650, 5479, 7770, 6165, 14358, 9458, 5851, 4373, 6399, 7308, 7852, 2918, 5821, 7179, 6353, 38491, 5779, 8652, 4192, 15714, 4157, 5805, 4970, 2262, 5703, 2855, 6434, 1673, 3334, 225, 5610, 2637, 24715, 6334, 6613, 1481, 4793, 198, 1954]

# Categories_stats =  [262465, 7113, 43867, 8725, 5135, 6069, 4571, 9973, 10759, 12884, 1865, 1983, 1285, 9838, 10806, 4768, 5508, 6587, 9509, 8147, 5513, 1294, 5303, 5131, 8720, 11431, 12354, 6496, 6192, 2682, 6646, 2685, 6347, 9076, 3276, 3747, 5543, 6126, 4812, 24342, 7913, 20650, 5479, 7770, 6165, 14358, 9458, 5851, 4373, 6399, 7308, 7852, 2918, 5821, 7179, 6353, 38491, 5779, 8652, 4192, 15714, 4157, 5805, 4970, 2262, 5703, 2855, 6434, 1673, 3334, 225, 5610, 2637, 24715, 6334, 6613, 1481, 4793, 198, 1954]

# # print(Categories_stats)
# sorted_categories_stats = sorted(Categories_stats, reverse=True)
# # print(sorted_categories_stats)

# plt.figure(figsize=(10, 5))
# plt.bar(range(len(sorted_categories_stats)), sorted_categories_stats)
# plt.xlabel("Categories")
# plt.ylabel("Num_objects")
# plt.show()