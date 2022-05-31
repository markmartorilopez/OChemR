# Copyright (c) 2022 rxn4chemistry - Mark Martori Lopez

# Augmentations
import albumentations as A
import cv2
import glob
import os
import numpy as np
import torchvision.transforms.functional as TF
import random

def my_rotation(image, bonding_box_coordinate):
    if random.random() > 0.5:
        angle = random.randint(-30, 30)
        image = TF.rotate(image, angle)
        bonding_box_coordinate = TF.rotate(bonding_box_coordinate, angle)
    # more transforms ...
    return image, bonding_box_coordinate

def own_transforms(crop,p_horiz,p_vert,p_bright,path):
    # In case crop can't be done...
    transform2 = A.Compose([
    A.HorizontalFlip(p=p_horiz),
    A.VerticalFlip(p=p_vert),
    A.RandomBrightnessContrast(p=p_bright),
    ], bbox_params=A.BboxParams(format='coco')) # albumeration (normalized and x,y,w,h)
    
    transform = A.Compose([
        A.RandomCrop(width=crop, height=crop),
        A.HorizontalFlip(p=p_horiz),
        A.VerticalFlip(p=p_vert),
        A.RandomBrightnessContrast(p=p_bright),
    ], bbox_params=A.BboxParams(format='coco')) # albumeration (normalized and x,y,w,h)

    c = 0
    for imgname in glob.glob(path):

        # Read Image
        image = cv2.imread(imgname)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height,width ,_= image.shape

        bboxes = []
        # Read labels from .txt and create new .txt file
        notcroped_txt_file = imgname.replace(".png",".txt")
        with open(notcroped_txt_file,"r") as file: # get text file with same name.
            newimgname = imgname.replace(".","_au.") # generate new name for png.
            newfilename = imgname.replace(".png","_au.txt") # generate new name for txt.
            with open(newfilename,"w") as newfile: # create new text file.
                for line in file.readlines():
                    obj = line.strip().split()

                    x = int(float(obj[1]) * width)
                    y = int(float(obj[2]) * height)
                    w = int(float(obj[3]) * width)
                    h = int(float(obj[4]) * height)
                    cls = str(obj[0])
                        
                    # Create bboxes in albumeration format
                    bboxes.append([x,y,w,h,cls])
                try:
                    if crop == 0:
                        transformed = transform2(image=image, bboxes = bboxes)
                    else:
                        
                        transformed = transform(image=image, bboxes=bboxes)

                    transformed_image = transformed['image']
                    transformed_bboxes = transformed['bboxes']        
                    cv2.imwrite(newimgname, transformed_image)
                    newimg = cv2.imread(newimgname)
                    height,width,_ = newimg.shape

                    # write bboxes and classes
                    for el in transformed_bboxes:
                        newfile.write(str(el[4]) + ' ' + str(float(el[0] / width)) + ' ' + str(float(el[1] / height)) + ' ' + str(float(el[2] / width)) + ' ' + str(float(el[3] / height))+'\n')
                    c+=1
                except: # If crop can't be done bc image w or h is smaller...
                    if os.path.exists(newfilename):
                        os.system("rm "+newfilename)
                    if os.path.exists(newimgname):
                        os.system("rm "+newimgname)
          
    print(f"{c} number of files with DA.")


# image_folder = "/dccstor/arrow_backup/images/train/"

# own_transforms(350, 0.5, 0.5, 0.2, image_folder + "*1.png")
# own_transforms(300,0.5,0.5,0.5,image_folder + "*2.png")
# own_transforms(500, 1, 1, 0.5, image_folder + "*3.png")
# own_transforms(400, 0.5, 1, 0.5, image_folder + "*4.png")
# own_transforms(600, 1, 0.5, 0.5, image_folder + "*5.png")
# own_transforms(700, 1, 1, 0.5, image_folder + "*6.png")