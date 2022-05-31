# Copyright (c) 2022 rxn4chemistry - Mark Martori Lopez

# Parameters
import json

# Loop
import os
from os import path
import time
import random

# Own functions
from functions.data_load import prepareMolecules
from functions.image_creation import finalImageCreation
import functions.logger as logger
from functions.logger import catch_and_log
from functions.data_aug.blurr_images import blurr_images
from functions.data_aug.coline_images import coline_images
from functions.data_aug.data_aug import own_transforms

# Parser
import argparse
from datetime import datetime

@catch_and_log
def runExperiment(loc,labelled_images):
    # Load data  -  Get a list of SMILES molecules.
    path = "data/molecules.txt"
    mol_SMILES = prepareMolecules(path)


    # Params  -  Read all parameters
    with open("params/params.json") as f:
        full_params = json.load(f)
        
    params = full_params["dataset_params"]

    train_path = params["train_path"]                   # Path to store training samples
    labelled_path = params["labelled_path"]             # Path to store labelled training samples.
    img_ws =  params["img_width"]                       # 1024 Image width
    img_hs =  params["img_height"]                      # 1024 Image height
    b_sizes =  params["molecules_sizes"]                # [30,45,60] # Size of chemical bonds in molecules.
    rotations =  params["molecules_rotations"]          # [0,30,330]
    n_molecules = params["num_molecules_per_reaction"]  # n molecules that will appear in each final image.
    n_samples = params["num_reactions_per_epoch"]       # number of samples created per each epoch (with its variables, rotation...)
    epochs = params["epochs"]                           # num of epochs


    # Print metadata
    print(f" @author: Mark Martori Lopez - IBM Research \n\n---- Syntetic dataset creation with the following params: ----")
    for param in params.keys():
        print(f"{param} = {params[param]}")

    print(f" -> Labelled images set to {labelled_images}")
    print()
    # RUN  -  LOOP
    total_it = 0
    for epoch in range(epochs):
        initial_time = time.time()
        img_w = img_ws[epoch]
        img_h = img_w
        rotation = random.sample(rotations,1)[0]
        # bond_size = random.sample(b_sizes,1)[0]
        bond_size = b_sizes[epoch]
        for it in range(0,n_samples):
            n_molec = random.sample(range(2,n_molecules+1),1)[0]
            img = finalImageCreation(train_path, labelled_path, n_molec, img_w, img_h, bond_size, rotation, it, epoch, mol_SMILES,labelled_images)
            if it % 250 == 0 and it != 0:
                time_now = time.time()
                print(f"Image {it}/{n_samples} of epoch {epoch} completed.", end = '\n', flush= True)
                print(f"{round(((total_it+it)/(n_samples * epochs))*100,2)}% of total task done in {round(abs(time_now - initial_time))} seconds.")   
        total_it += n_samples
        tf = time.time()
        print(f"Epoch {epoch+1}/{epochs} completed in {round(abs(tf-initial_time))} seconds.", end = '\n', flush=True)
        


loc_parser = argparse.ArgumentParser()
loc_parser.add_argument("-l", "--log-folder" , help="Folder to which to save logs", action='store', default='logs')
loc_parser.add_argument("-p", "--params-folder" , help="Folder that contains json file with params", action='store')
loc_parser.add_argument("-cr", "--crop" , help="Cropping Data Augmentation", default = (0,0))
loc_parser.add_argument("-ph", "--p-horiz" , help="Probability of Horizontal flip", default = 0)
loc_parser.add_argument("-pv", "--p-vert" , help="Probability of Vertical flip", default = 0)
loc_parser.add_argument("-pbr", "--p-bright" , help="Probability of brightness change", default = 0)
loc_parser.add_argument("-b", "--blurred" , help="add blurred Data Augmentation", default = False)
loc_parser.add_argument("-sk", "--skeletonization" , help="add blurred Data Augmentation", default = False)
loc_parser.add_argument("-cl", "--colored_lines" , help="add coloured blobs Data Augmentation", default = False)
loc_parser.add_argument("-li", "--labelled-images" , 
            help="store labelled images with drawn rectangles. Atention: Big load memory.", 
            default= False) # action = 'store_false', default= False)

file_arg = loc_parser.parse_args()


if __name__ == "__main__":

    # # Logger
    now = datetime.now()
    loc = file_arg.log_folder + '/' + now.strftime("%b-%d-%Y %H:%M:%S") # 'logs' file_arg.folder # Location of folder that contains the json
    os.mkdir(loc)
    logger.init_logger(loc)

    # Labelled Images (Yes/No):
    labelled_images = file_arg.labelled_images

    runExperiment(loc, labelled_images)

    # DATA AUGMENTATION 
    blurred = file_arg.blurred
    skeletonization = file_arg.skeletonization
    colored_lines = file_arg.colored_lines
    crop = file_arg.crop
    p_horiz = file_arg.p_horiz
    p_vert = file_arg.p_vert
    p_bright = file_arg.p_bright

    ## TO USE only if DL Model has no Augmentations.
    # if "train" in train_path:

    #     own_transforms(crop, p_horiz, p_vert, p_bright, train_path + "*1.png")
    #     own_transforms(300,0.5,0.5,0.5,train_path + "*2.png")
    #     own_transforms(500, 1, 1, 0.5, train_path + "*3.png")
    #     own_transforms(400, 0.5, 1, 0.5, train_path + "*4.png")
    #     own_transforms(600, 1, 0.5, 0.5, train_path + "*5.png")
    #     own_transforms(700, 1, 1, 0.5, train_path + "*6.png")
    
    print("Job finished! Well done :)")
