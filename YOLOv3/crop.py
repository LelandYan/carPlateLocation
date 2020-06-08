import cv2

img = cv2.imread("0204.jpg")
print(img.shape)
cropped = img[106:141, 102:218]  # 裁剪坐标为[y0:y1, x0:x1] 133,181,307,233
cv2.imwrite("cv_cut_thor.jpg", cropped)
