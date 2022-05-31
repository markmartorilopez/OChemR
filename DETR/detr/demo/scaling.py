import cv2
import torch

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x[0],x[1],x[2],x[3]
    b = ((x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h))
    return b


def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = (round(b[0]/img_w,4), round(b[1]/img_h, 4), round(b[2]/img_w, 4),round(b[3]/img_h, 4))
    return b

image = 'image.png'
img = cv2.imread(image)
wi,he,_ = img.shape
with open('not_scaled.txt', 'w') as output:
    with open('labels.txt','r') as f:
            for line in f.readlines():
                obj = line.strip().split()
                # x = int(float(obj[1]) * wi)
                # y = int(float(obj[2]) * he)
                # w = int(float(obj[3]) * wi)
                # h = int(float(obj[4]) * he)
                size = (wi,he)
                bbox = [float(obj[1]),float(obj[2]),float(obj[3]),float(obj[4])]

                b = rescale_bboxes(bbox,size)
                
                output.write('0' + ' ' + str(b[0]) + ' ' + str(b[1]) + ' ' + str(b[2]) + ' ' + str(b[3]) + '\n')

                cv2.rectangle(img,(int(b[0]), int(b[1])),(int(b[2]),int(b[3])), color=(255,0,100), thickness = 2)
                cv2.imwrite('2image_labelled.png',img)