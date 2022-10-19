import cv2
import numpy as np
import math
import console

def preprocess(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 1) # Gaussian Filter
    ret, img_thr = cv2.threshold(img_blur, 70, 255, cv2.THRESH_BINARY) # Makes picture clearer
    img_cany = cv2.Canny(img_thr, 50, 50) # Edge detection
    kernel = np.ones((3,3), np.uint8)
    img_dilate = cv2.dilate(img_cany, kernel, iterations = 2) # Edge expansion
    img_erode = cv2.erode(img_dilate, kernel, iterations = 1) # Edge corrosion

    # cv2.imwrite("../arrows/res_erode_1.png", img_erode) 
    return img_erode

def find_tip(points, convex_hull):
    length = len(points)
    indices = np.setdiff1d(range(length), convex_hull) # Find index of 1 points in concave of arrow
    for i in range(2):
        j = indices[i] + 2
        if j > length - 1:
            j = length - j
        p = j + 2
        if p > length - 1:
            p = length - p
        if np.all(points[j] == points[indices[i-1]-2]):
            return tuple([points[j], points[p]])

def rotate(angle, xy):
    rotatex = math.cos(angle)*xy[0] - math.sin(angle)*xy[1]
    rotatey = math.cos(angle)*xy[0] + math.sin(angle)*xy[1]
    return tuple([rotatex,rotatey])

img = cv2.imread("../arrows/1.png")
w,h,c = img.shape

cnts,hierarchy = cv2.findContours(preprocess(img), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE) # Get image outline.
for cnt in cnts:
    peri = cv2.arcLength(cnt, True) # Get perimeter
    approx = cv2.approxPolyDP(cnt, 0.025 * peri, True) # Get endpoint and return polygon endopoint value of the contour.
    hull = cv2.convexHull(approx, returnPoints = False) # Get the covnex hull and return the index of the convex hull corner.

    sides = len(hull)
    if 6 > sides > 3 and sides + 2 == len(approx):
        arrow_tip = find_tip(approx[:,0,:], hull.squeeze())
    else:
        arrow_tip = 0
    # Judge direction:
    if arrow_tip:
        arrow_dir = np.array(arrow_tip[0]) - np.array(arrow_tip[1])
        arrow_rotate = rotate(math.pi/4, arrow_dir)

        if arrow_rotate[0] > 0:
            if arrow_rotate[1] > 0:
                print("Right")
            else:
                print("Up")
        else:
            if arrow_rotate[1] > 0:
                print("Down")
            else:
                print("Left")


            