# Copyright (c) 2022 rxn4chemistry - Mark Martori Lopez

# Image
import cv2
# Loop
import random
import numpy as np
# Image
import cv2
from PIL import Image
from PIL import ImageDraw, ImageFont
# Created files
from functions.classes.drawings import Molecule, Symbol, WhiteBackground
from functions.labellingImage import writeLabelsInFile
from functions.logger import catch_and_log
from functions.data_aug.skeletonization import skelet_images

# image_folder = "/dccstor/arrow_backup/images/train/"
# image_folder = "results/images/"

def addColouredBoxes(img, current_width, current_height, step, molecules_SMILES):
    """
    Adds Coloured Boxes below certain arrows randomly.
    """
    max_w, max_h = img.size
  
    # Create coloured Box: Pasting images
    ypos = int((max_h / 2) + int(max_h * 0.164)) # y coord for coloured box with mol inside.
    long_y = int(max_w * 0.45) # random.randint(int(max_w * 0.45),int(max_w * 0.47)) # Longitude of rectangle, give 4 as padding if maximum size chosen...
    long_x = int(max_w * 0.41) # random.randint(int(max_w * 0.41),int(max_w * 0.58))
    xpos = 1 # random.randint(int(max_w * 0.12),int(max_w * 0.24))

    # Get coordinates - Coloured Boxes
    coords = []
    x_coord_cb = current_width + xpos # coordinates of Coloured boxes
    y_coord_cb = ypos + current_height
    cb_coord_w = x_coord_cb + int(0.42 * max_w) 

    cb_coord_h = y_coord_cb + int(0.25 * max_h) # 72
    coords.append(['0', x_coord_cb, y_coord_cb, cb_coord_w, cb_coord_h])

    # # Draw a rectangle into arrow image and GET COORDINATES
    rect = ImageDraw.Draw(img)  
    # # rect.rectangle([(xpos,ypos),(max_w - 20,max_h - 5)],outline=(255,0,255))
    rect.rectangle([(xpos,ypos),(xpos + int(0.33 * max_h),ypos + int(0.33 * max_h))],outline=(255,255,255))

    molecr = random.sample(molecules_SMILES,1)[0]
    pos = molecules_SMILES.index(molecr)
    # mol_name = molecules_names[pos]

    # Get new molecule
    ch,im2, crds = Molecule(long_x - int(max_w * 0.06),long_y - int(max_w * 0.06), 15,20).drawI(molecr,
                                                        'v',current_width, current_height, step)
    # Paste the molecule to the image with the rectangle.
    img.paste(im2,(xpos+int(max_w * 0.05),ypos+int(max_w * 0.03)))

    return img, coords

def finalImageCreation(train_path,labelled_path, n_mol, reaction_w, reaction_h, bond_size, rotation, it, ep, molecules_SMILES, labelled_images):
    """
    Builds the whole path of the reaction and returns the final image.

    I:
        combinations : list
                List of molecules to be used in the reaction.
    O:
        final_image : img
                Final image created

    reaction positions  = {1 2 3 4 5 6
                          12 11 10 9 8 7
                          13 14 15 16 17 18
                          24 23 22 21 20 19}
    """
    # Chose randomly which n_mol molecules to use:
    used_molecules = list(random.sample(molecules_SMILES,n_mol + 10))
    num_molecules_used = 0
    # Letters
    al = 'ABCDEFGIJKLMNPRSTUVWXYZ123456789'  # Max 12 molecules in one image. IMPROVE 1 - ADD ALL.
    white_img_path = 'symbols/white.png'     # White image path

    # Dictionaries used:
    reaction = {}           # Stores sub-images to create final image.
    coordinates = {}        # Stores coordinates of bboxes in labelled images.
    sizes = {}              # Stores where are we in each step.

    # Variables
    total_width = reaction_w
    total_height = reaction_h
    img_h = int(total_height / 4)
    current_width = 0 # start at left corner   
    current_height = 0 # start at top corner   left-top
    prev_draw = 'None' # Keep track if we need a molecule or a symbol.

    # Choose number of rows per image (depend on num of molecules)
    if n_mol < 4:
        images_ratio = 7
    elif n_mol < 7:
        images_ratio = 13
    elif n_mol < 10:
        images_ratio = 19
    else:
        images_ratio = 25

    for step in range(1,images_ratio): # 25

        if step % 6 == 0: # if we are at the most-right position
            img_w = int(total_width - current_width)    # get rest of width as img_w
        elif step in [7,13,19]:
            img_w = random.randint(int(0.1564 * total_width), int(0.167 * total_width)) # if total_width = 1024 -> [160-185]
            current_width = 0                           # set current width to 0. To start from left again.
            current_height += img_h
        else:
            # Determine size of each sub_image (molecules and arrows)
            img_w = random.randint(int(0.1564 * total_width), int(0.167 * total_width)) # if total_width = 1024 -> [160-185]

        if num_molecules_used >= n_mol: # FILL REST OF IMAGE WITH MOL or TEXT.
            if random.randint(1,2) == 1:
                prev_draw,reaction[step], coordinates[step] = WhiteBackground(img_w,img_h).drawI(white_img_path,current_width, current_height, "text")
            else:
                cur_mol = used_molecules[num_molecules_used]
                letter = random.sample(al,1)[0]
                prev_draw, reaction[step], coordinates[step] = Molecule(
                                img_w,img_h, bond_size,rotation).drawI(cur_mol,letter, 
                                        current_width, current_height, step)
                num_molecules_used +=1
        else:
            if step in [6,18]:
                prev_draw, reaction[step], coordinates[step] = Symbol(img_w,img_h,reaction_w).drawI(white_img_path, current_width, current_height, 'right')
            elif step in [7,19]:
                prev_draw, reaction[step], coordinates[step] = Symbol(img_w,img_h,reaction_w).drawI(white_img_path, current_width, current_height,'left')
            else:
                if prev_draw != 'molecule' or step == 13: # We need to draw a Mol
                    cur_mol = used_molecules[num_molecules_used]
                    letter = random.sample(al,1)[0]
                    print(f"Starting molecule sizes = {img_w,img_h}")
                    prev_draw, reaction[step], coordinates[step] = Molecule(
                                    img_w,img_h, bond_size,rotation).drawI(cur_mol,letter, 
                                    current_width, current_height, step)
                    num_molecules_used += 1                
                else: # We need to draw an arrow most of times.
                    if random.randint(1,50) == 1:
                        cur_mol = used_molecules[num_molecules_used]
                        letter = random.sample(al,1)[0]
                        prev_draw, reaction[step], coordinates[step] = Molecule(
                                        img_w,img_h, bond_size,rotation).drawI(cur_mol,letter, 
                                        current_width, current_height, step)
                        num_molecules_used += 1
                    else:
                        # If we are not in a corner...
                        direction = 'left' if random.randint(1,2) == 1 else "right"
                        if random.randint(1,8) == 1: # 1/8 times adds the "+" symbol.
                            prev_draw, reaction[step], coordinates[step] = Symbol(img_w,img_h,reaction_w).drawI("plus", current_width, current_height, direction)
                        else:
                            prev_draw, reaction['arrow_'+str(step)], coordinates[step] = Symbol(img_w,img_h,reaction_w).drawI(white_img_path, current_width, current_height, direction)

        sizes[step] = [current_width, current_height]
        current_width += img_w
        try:
            iimmgg = reaction[step]
        except:
            iimmgg = reaction['arrow_'+str(step)]
        print(f"Current Image Size = {iimmgg.size} type of img = {prev_draw}")

    # concatenate all the images and add coloured boxes - Improve...
    save_odd_row = [25,26,27,28,29,30] # Add coloured boxes in first or 3rd row of main image.
    for paso, key in enumerate(reaction.keys(), start = 1):
        if paso not in [7,8,9,10,11,12,19,20,21,22,23,24]:
            if isinstance(key,str):
                if random.randint(1,4) == 1:
                    # w,h = reaction[key].size
                    editedImage, coords = addColouredBoxes(reaction[key], sizes[paso][0], sizes[paso][1], paso, molecules_SMILES)
                    reaction[key] = editedImage
                    step = save_odd_row.pop(0)# Add cbox coordinates at the end of the dictionary
                    coordinates[step] = coords
                        
    all_images = list(reaction.values())
    row1 = all_images[0:6]
    hor1 = np.hstack((row1))

    if images_ratio < 8:
        final_image = hor1

    if images_ratio < 14 and images_ratio > 8:
        row2 = all_images[6:12]
        hor2 = np.hstack((row2))
        final_image = np.vstack((hor1,hor2))

    if images_ratio < 20 and images_ratio > 15:
        row2 = all_images[6:12]
        hor2 = np.hstack((row2))
        row3 = all_images[12:18]
        hor3 = np.hstack((row3))
        final_image = np.vstack((hor1,hor2,hor3))

    if images_ratio < 26 and images_ratio > 20:
        row2 = all_images[6:12]
        hor2 = np.hstack((row2))
        row3 = all_images[12:18]
        hor3 = np.hstack((row3))
        row4 = all_images[18:24]
        hor4 = np.hstack((row4))
        final_image = np.vstack((hor1,hor2,hor3,hor4)) # ,hor3,hor4))
    
    # store unlabelled images
    cv2.imwrite(train_path + 'Final_reaction_Epoch'+str(ep)+'_it_'+str(it)+'.png', final_image)
    # prepare coordinates, draw rectangles and create text file
    writeLabelsInFile(train_path, labelled_path, ep,it,coordinates,prev_draw,final_image, labelled_images)

    # Get skeletonizated image: In case of B&W
    # skelet_images(train_path + 'Final_reaction_Epoch'+str(ep)+'_it_'+str(it)+'.png')
