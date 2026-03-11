import cv2
import numpy as np

path = 'C:\\Users\\uig29841\\Pant_CVLearning\\ComputrvisionLearning-1\\edge_detection\\edges_2.png'

img = cv2.imread(path)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

edges = cv2.Canny(img_gray, 50, 150)
cv2.imshow('Edge Detected Image', edges)

corner = cv2.goodFeaturesToTrack(edges, 100, 0.04, 10)
corner = np.int_(corner)

gray = np.float32(img_gray)
corner2 = cv2.cornerHarris(gray, 2, 3, 0.07)
# cv2.imshow('Corners Detected', img)
# cv2.waitKey(0)

corner2 = cv2.dilate(corner2, None)
img[corner2 > 0.01 * corner2.max()] = [0, 255, 0]

for i in corner:
    x,y = i.ravel()
    cv2.circle(img, (x,y), 3, 255, -1)

cv2.imshow('dst',img)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()


cv2.imwrite('corners_output.png', img)

# cv2.imshow('Original Image', img)ll
# cv2.waitKey(0)
# cv2.imshow('Grayscale Image', img_gray)
# cv2.waitKey(0)

# cv2.imshow('Edge Detected Image', edges)
# cv2.blur(edges, (5,5))
# cv2.waitKey(0)
# cv2.imwrite('edges_output.png', edges)
# cv2.waitKey(0)
cv2.destroyAllWindows()
