# Copyright (c) 2022 rxn4chemistry - Mark Martori Lopez

import cv2
import numpy as np
import random
from PIL import Image, ImageDraw


#### RESCALING from int to [0,1]
def box_cxcywh_to_xyxy(x,y,w,h):
    x_c, y_c, w_, h_ = x,y,w,h
    b = ((x_c - 0.5 * w_), (y_c - 0.5 * h_),
         (x_c + 0.5 * w_), (y_c + 0.5 * h_))
    return b

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = (round(b[0]/img_w,4), round(b[1]/img_h, 4), round(b[2]/img_w, 4),round(b[3]/img_h, 4))
    return b


def labellingImage(image_p, cls, current_width, current_height, step):
    """
    Finds x, y, width and height coordinates of bounding boxes for each molecule in image.

    I: 
        image : opencv image
                image to be labelled.
    O:
        coords : list
                coords in final image with the molecule's loc. 
    """
    image = cv2.imread(image_p)

    # Find coordinates of all pixels below threshold
    coords = np.argwhere(image[:,:,:] != 255 )
    if len(coords) < 1:
        return 'Nothing'
    
    y = np.amin(coords, axis=0)[0]
    x = np.amin(coords, axis=0)[1]
    
    h = np.amax(coords, axis=0)[0]
    w = np.amax(coords, axis=0)[1]

    coords = [cls,x+current_width,y+current_height,current_width + w, current_height + h]
    
    return coords

def writeLabelsInFile(train_path,labelled_path, ep,it,coordinates,prev_draw,final_image, labelled_images):
    """
    Get coordinates from dictionary and write its correct form in the labeled text output file.
    """
    # Open a file
    labelling = random.randint(1,100) # Store only 1 out of 100 labelled imgs.
    with open(train_path + 'Final_reaction_Epoch'+str(ep)+'_it_'+str(it)+'.txt','w') as coordinates_file:
        for coords in list(coordinates.keys()):
            try:
                height,width,_ = final_image.shape
            except:
                height,width = final_image.shape
            # print(F"Coordinates of reaction {coords} = {coordinates[coords]}\n")
            for each in coordinates[coords]:
                # Colors: '0'molecules green, '1'arrows red, '2' and '3' text in gray and '4'colouredbox
                color_rect = (169,169,169) if each[0] in ['2'] else (255,64,64) # gray, red
                color_rect = (0,0,250) if each[0] in '3' else color_rect # blue
                if each[0] == '0': # green
                    color_rect = (102,205,0)
                    
                x = each[1]
                y = each[2]
                w = abs(x - each[3])
                h = abs(y - each[4])

                if labelled_images:
                    if labelling == 1:
                        # I = ImageDraw.Draw(final_image)
                        # I.rectangle((each[1], each[2]),(each[1] + w ,each[2] + h), outline = color_rect, width = 1)
                        cv2.rectangle(final_image,(each[1], each[2]),(each[1] + w ,each[2] + h), color=color_rect, thickness = 1)
                
                x,y,w,h = x/width, y/height ,w/width, h/height
                
                coordinates_file.write(each[0] + ' ' + str(x) + ' ' + str(y) + ' ' + str(w) + ' ' + str(h) + '\n')
                        
        if labelled_images:
            if labelling == 1:
                # final_image.save(labelled_path+'Final_reaction_Epoch'+str(ep)+'_it_'+str(it)+'labelled.png')
                cv2.imwrite(labelled_path+'Final_reaction_Epoch'+str(ep)+'_it_'+str(it)+'labelled.png', final_image)
                        