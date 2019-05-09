import sys

import dlib
import numpy as np
from skimage import io
import os
path = '../web_crawler/resource/getchu_raw_img'
files= os.listdir(path) #得到文件夹下的所有文件名称
# print(files)
# print(len(files))
# files = ['1100_0.jpg']

# 使用dlib自带的frontal_face_detector作为我们的特征提取器
detector = dlib.get_frontal_face_detector()

# 使用dlib提供的图片窗口
win = dlib.image_window()
path_save = "./getchu_detector/"
# sys.argv[]是用来获取命令行参数的，sys.argv[0]表示代码本身文件路径，所以参数从1开始向后依次获取图片路径
for f in files:
    # 输出目前处理的图片地址
    print("Processing file: {}".format(f))

    # 使用skimage的io读取图片
    image_path = path + '/'+ f
    img = io.imread(image_path)

    # 使用detector进行人脸检测 dets为返回的结果
    dets = detector(img, 1)
    # dets的元素个数即为脸的个数
    print("Number of faces detected: {}".format(len(dets)))

    # 使用enumerate 函数遍历序列中的元素以及它们的下标
    # 下标i即为人脸序号
    # left：人脸左边距离图片左边界的距离 ；right：人脸右边距离图片左边界的距离
    # top：人脸上边距离图片上边界的距离 ；bottom：人脸下边距离图片上边界的距离
    for i, d in enumerate(dets):
        print("dets{}".format(d))
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}"
              .format(i, d.left(), d.top(), d.right(), d.bottom()))
        # # 读取人脸矩形的坐标
        # pos_start = tuple([d.left(), d.top()])
        # pos_end = tuple([d.right(), d.bottom()])

        # 计算人脸矩形大小
        height = (d.bottom() - d.top())
        width = (d.right() - d.left())
        height1 = int(height*1.5)
        width1 = int(width*1.5)

        # 生成新的空白图像
        img_blank = np.zeros((height1, width1, 3), np.uint8)

        # 将人脸写入空白图像
        for m in range(height1):
            for n in range(width1):
                top = d.top() - int(height/2)
                left = d.left() -int(width/2)
                img_blank[m][n] = img[top + m][left + n]

        # 保存人脸到本地
        print("Save to:", path_save + f[:-4] + ".jpg")
        io.imsave(path_save + f[:-4] + ".jpg", img_blank)
  #   # 也可以获取比较全面的信息，如获取人脸与detector的匹配程度
  #   dets, scores, idx = detector.run(img, 1)
  #   for i, d in enumerate(dets):
  #       print("Detection {}, dets{},score: {}, face_type:{}".format(i, d, scores[i], idx[i]))
  #
  #       # 绘制图片(dlib的ui库可以直接绘制dets)
  #   win.set_image(img)
  #   win.add_overlay(dets)
  #
    # win.set_image(img)
    # win.set_image(img_blank)
  #等待点击
    # dlib.hit_enter_to_continue()

