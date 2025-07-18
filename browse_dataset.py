# 使用方法：
# 1、修改输入图片文件夹
# 2、修改输入标签文件夹
# 3、输出图片文件夹位置默认为本程序所在文件夹，名称为output
# 4、修改标签类别
# 注意，本代码只适用于yolo格式的标签
import os
import numpy as np
import cv2
 
# 修改输入图片文件夹
img_folder = "../datasets/TrashCAN_material/images/val"
img_list = os.listdir(img_folder)
img_list.sort()
# 修改输入标签文件夹
label_folder = "../datasets/TrashCAN_material/labels/val"
label_list = os.listdir(label_folder)
label_list.sort()
# 输出图片文件夹位置
path = os.getcwd()
output_folder = path + '/' + str("browse_dataset_output")
if not os.path.exists(output_folder):
    os.mkdir(output_folder)

# DUO
# # 修改你的类别
# labels = ['holothurian', 'echinus', 'scallop','starfish']  #在此处修改你的类别
# # 不同类别的颜色
# colormap = [(231, 239, 250), (209, 73, 91), (102, 179, 76), (128, 0, 128)]  # 色盘，根据类别添加颜色

# VOC 
# labels = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
#           'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
# colormap = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0), (0, 0, 0), (255, 255, 255),
#            (128, 128, 0), (128, 0, 128), (0, 128, 128), (128, 128, 128), (128, 0, 0), (0, 128, 0), (0, 0, 128), (0, 128, 128), (0, 255, 128),
#            (128, 255, 0), (128, 255, 128), (128, 255, 255)]

# TrashCAN material version
labels = [
  "rov",
  "plant",
  "animal_fish",
  "animal_starfish",
  "animal_shells",
  "animal_crab",
  "animal_eel",
  "animal_etc",
  "trash_etc",
  "trash_fabric",
  "trash_fishing_gear",
  "trash_metal",
  "trash_paper",
  "trash_plastic",
  "trash_rubber",
  "trash_wood",
]
colormap = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0), (0, 0, 0), (255, 255, 255),
            (128, 128, 0), (128, 0, 128), (0, 128, 128), (128, 128, 128), (128, 0, 0), (0, 128, 0), (0, 0, 128), (0, 128, 128),
]
 
# Brackish/images/test
# labels = [
# 'fish',
# 'small_fish',
# 'crab',
# 'shrimp',
# 'jellyfish',
# 'starfish',
# ]
# colormap = [
# (0, 0, 255),
# (0, 255, 0),
# (255, 0, 0),
# (0, 255, 255),
# (255, 0, 255),
# (255, 255, 0),
# ]
 
# 坐标转换
def xywh2xyxy(x, w1, h1, img):
    label, x, y, w, h = x
    # print("原图宽高:\nw1={}\nh1={}".format(w1, h1))
    # 边界框反归一化
    x_t = x * w1
    y_t = y * h1
    w_t = w * w1
    h_t = h * h1
    # print("反归一化后输出：\n第一个:{}\t第二个:{}\t第三个:{}\t第四个:{}\t\n\n".format(x_t, y_t, w_t, h_t))
    # 计算坐标
    top_left_x = x_t - w_t / 2
    top_left_y = y_t - h_t / 2
    bottom_right_x = x_t + w_t / 2
    bottom_right_y = y_t + h_t / 2
 
    # 绘制矩形框，使用与类别相关的颜色
    class_color = colormap[int(label)]
    cv2.rectangle(img, (int(top_left_x), int(top_left_y)), (int(bottom_right_x), int(bottom_right_y)), class_color, 2)
 
    # 在矩形框上方添加类别信息
    class_name = labels[int(label)]
    text = f"{class_name}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_x = int(top_left_x)
    text_y = int(top_left_y) - 5  # 调整文本位置
    cv2.putText(img, text, (text_x, text_y), font, font_scale, class_color, font_thickness)
 
    return img
 
 
if __name__ == '__main__':
    print("len(img_list):",len(img_list))
    print("len(label_list):",len(label_list))
    for i in range(len(label_list)):
        image_path = img_folder + "/" + img_list[i]
        label_path = label_folder + "/" + label_list[i]

        #读取图像文件
        img = cv2.imread(str(image_path))
        h, w = img.shape[:2]
        # 读取 labels
        with open(label_path, 'r') as f:
            lb = np.array([x.split() for x in f.read().strip().splitlines()], dtype=np.float32)
        # 绘制每一个目标
        for x in lb:
            # 反归一化并得到左上和右下坐标，画出矩形框并添加类别信息
            img = xywh2xyxy(x, w, h, img)
        cv2.imwrite(output_folder + '/' + '{}.png'.format(image_path.split('/')[-1][:-4]), img)