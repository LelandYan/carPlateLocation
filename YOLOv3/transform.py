import os
import json
import cv2 as cv
import numpy as np

fileList = os.listdir()
fileList.remove(".idea")
fileList.remove("via_region_data.json")
fileList.remove("test.py")
fileList.remove("position.py")
fileList.remove("clips.py")
# fileList = [file.split(".")[0] for file in fileList]
# print(fileList)
# print(fileList)
fileName = "via_region_data.json"
# 设置序号
cnt = 1
"""
每行格式如下：
class_index center_x cnter_y w h
每行五个字段，用空格分开。第一个字段class_index是分类的序号，从数字0开始，对应class.names里的顺序。
center_x和cnter_y是bbox中心的坐标，w和h是bbox的宽和高。注意着四个值都是小于1的浮点数，center_x和w是绝对值除以图片宽度得到的，
cnter_y和h是绝对值除以图片高度得到的。
"""
with open(fileName, 'r', encoding='utf-8') as f:
    popData = json.load(f)
    dictKey = list(popData.keys())
    for key in dictKey:
        fileNum = popData[key]["filename"]
        allPointsX = popData[key]["regions"][0]["shape_attributes"]["all_points_x"]
        allPointsY = popData[key]["regions"][0]["shape_attributes"]["all_points_y"]
        img = cv.imdecode(np.fromfile(fileNum, dtype=np.uint8), cv.IMREAD_COLOR)
        height = img.shape[0]
        width = img.shape[1]
        minX = min(allPointsX)
        maxX = max(allPointsX)
        minY = min(allPointsY)
        maxY = max(allPointsY)
        w = maxX - minX
        h = maxY - minY
        centerX = (minX + maxX) * 1.0 / 2
        centerY = (minY + maxY) * 1.0 / 2
        os.rename(fileNum, f"{cnt}.jpg")
        b = [centerX / width, centerY / height, w / width, h / height]
        with open(f"{cnt}.txt", 'w') as f2:
            f2.write(str(0) + " " + " ".join([str(a) for a in b]) + "\n")
        print(fileNum, allPointsX, allPointsY)
        cnt += 1

# for child in children:
#     if child[0].tag == "imageName" and child[0].text in fileList:
#         x = int(child[1][0].attrib['x'])
#         y = int(child[1][0].attrib['y'])
#         width = int(child[1][0].attrib['width'])
#         height = int(child[1][0].attrib['height'])
#         b = (int(x - width), int(y - height), int(x + width), int(y + height))
#         f = open(f"{cnt}.txt", 'w')
#         f.write(str(0) + " " + ",".join([str(a) for a in b]) + "\n")
#         f.close()
#         print(child[0].text + ".jpg" + "->" + f"{cnt}.jpg")
#         os.rename(child[0].text + ".jpg", f"{cnt}.jpg")
#         cnt += 1
#         # print(child[1].attribute["x"])
