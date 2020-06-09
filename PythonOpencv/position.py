import cv2 as cv
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
import sys, os, json, random


class LPRAlg:
    maxLength = 700
    minArea = 2000

    def __init__(self, imgPath=None):
        self.imgOri = cv.imread(imgPath)
        if self.imgOri is None:
            print("Cannot load this picture!")
            raise Exception
        self.colorList = []
        self.imgPlatList = []

    def findVehiclePlate(self):

        def pointLimit(point, maxWidth, maxHeight):
            if point[0] < 0:
                point[0] = 0
            if point[0] > maxWidth:
                point[0] = maxWidth
            if point[1] < 0:
                point[1] = 0
            if point[1] > maxHeight:
                point[1] = maxHeight

        # Step1: Resize
        img = np.copy(self.imgOri)
        h, w = img.shape[:2]
        # 将图片剪裁为固定大小
        if w > self.maxLength:
            resizeRate = self.maxLength / w
            img = cv.resize(img, (self.maxLength, int(h * resizeRate)), interpolation=cv.INTER_AREA)
            w, h = self.maxLength, int(h * resizeRate)
        imgWidth, imgHeight = w, h
        cv.imwrite("result1.jpg", img)
        cv.imshow("imgResize", img)

        # Step2: Prepare to find contours
        # 双边滤波
        # img1 = cv.bilateralFilter(img, 9, 75, 75)
        # cv.imshow("img1", img1)
        # 高斯滤波
        img = cv.GaussianBlur(img, (3, 3), 0)
        imgGary = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # cv.imshow("imgGary", imgGary)
        # 全局直方图均衡化
        # dst = cv.equalizeHist(imgGary)

        # 自适应直方图均衡化
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        imgGary = clahe.apply(imgGary)
        # cv.imshow("dst", dst)
        # cv.imshow("dst1", imgGary)
        kernel = np.ones((20, 20), np.uint8)
        imgOpen = cv.morphologyEx(imgGary, cv.MORPH_OPEN, kernel)
        # cv.imshow("imgOpen", imgOpen)

        imgOpenWeight = cv.addWeighted(imgGary, 1, imgOpen, -1, 0)
        # cv.imshow("imgOpenWeight", imgOpenWeight)

        ret, imgBin = cv.threshold(imgOpenWeight, 0, 255, cv.THRESH_OTSU + cv.THRESH_BINARY)
        # cv.imshow("imgBin", imgBin)

        imgEdge = cv.Canny(imgBin, 100, 200)
        # cv.imshow("imgEdge", imgEdge)

        kernel = np.ones((4, 19), np.uint8)
        imgEdge = cv.morphologyEx(imgEdge, cv.MORPH_CLOSE, kernel)
        imgEdge = cv.morphologyEx(imgEdge, cv.MORPH_OPEN, kernel)
        # cv.imshow("imgEdgeProcessed", imgEdge)

        # Step3: Find Contours
        image, contours, hierarchy = cv.findContours(imgEdge, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        contours = [cnt for cnt in contours if cv.contourArea(cnt) > self.minArea]

        # Step4: Delete some rects
        carPlateList = []
        imgDark = np.zeros(img.shape, dtype=img.dtype)
        for index, contour in enumerate(contours):
            rect = cv.minAreaRect(contour)  # [中心(x,y), (宽,高), 旋转角度]
            w, h = rect[1]
            if w < h:
                w, h = h, w
            scale = w / h
            if 2 < scale < 5.5:
                # color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                color = (255, 255, 255)
                carPlateList.append(rect)
                cv.drawContours(imgDark, contours, index, color, 1, 8)

                box = cv.boxPoints(rect)  # Peak Coordinate
                box = np.int0(box)
                # Draw them out
                cv.drawContours(imgDark, [box], 0, (0, 0, 255), 1)

        # cv.imshow("imgGaryContour", imgDark)
        print("Vehicle number: ", len(carPlateList))

        # Step5: Rect rectify
        imgPlats = []
        for index, carPlat in enumerate(carPlateList):
            if -1 < carPlat[2] < 1:
                angle = 1
            else:
                angle = carPlat[2]
            carPlat = (carPlat[0], (carPlat[1][0] + 5, carPlat[1][1] + 5), angle)
            box = cv.boxPoints(carPlat)

            # Which point is Left/Right/Top/Bottom
            w, h = carPlat[1][0], carPlat[1][1]
            if w > h:
                LT = box[1]
                LB = box[0]
                RT = box[2]
                RB = box[3]
            else:
                LT = box[2]
                LB = box[1]
                RT = box[3]
                RB = box[0]

            for point in [LT, LB, RT, RB]:
                pointLimit(point, imgWidth, imgHeight)

            # Do warpAffine
            newLB = [LT[0], LB[1]]
            newRB = [RB[0], LB[1]]
            oldTriangle = np.float32([LT, LB, RB])
            newTriangle = np.float32([LT, newLB, newRB])
            warpMat = cv.getAffineTransform(oldTriangle, newTriangle)
            imgAffine = cv.warpAffine(img, warpMat, (imgWidth, imgHeight))
            # cv.imshow("imgAffine" + str(index), imgAffine)

            imgPlat = imgAffine[int(LT[1]):int(newLB[1]), int(newLB[0]):int(newRB[0])]
            imgPlats.append(imgPlat)
            cv.imwrite("result1cut.jpg", imgPlat)
            cv.imshow("imgPlat" + str(index), imgPlat)


if __name__ == '__main__':
    L = LPRAlg("197.jpg")
    L.findVehiclePlate()
    cv.waitKey(0)
