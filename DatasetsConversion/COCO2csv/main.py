''' This file is used to convert annotations from .txt file to tenforflow csv formate
    Command : python main.py -i /dccstor/arrow_backup/images/train/ -img /dccstor/arrow_backup/images/train/ -o train.csv
    
    Output format will be:
    filename ,height ,width ,class ,xmin ,ymin ,w ,h 

'''

import os
import os.path
import argparse
import pandas as pd
from PIL import Image
from tqdm import tqdm
import glob


def write_to_csv(ann_path ,img_path ,dict):
    annos = []
    # Read txts  
    for ima in tqdm(glob.glob(img_path+"*.png")):
        # Read image and get its size attributes
        im = Image.open(ima)
        wi = im.size[0]
        he = im.size[1]

        # Read txt file 
        with open(ima.replace("png","txt"), "r") as filelabel:
            for line in filelabel.readlines():
                obj = line.strip().split()

                cls = obj[0]

                xmin = int(float(obj[1]) * wi)

                ymin = int(float(obj[2]) * he)

                w = int(float(obj[3]) * wi) - 1
                h = int(float(obj[4]) * he) - 1


                annos.append([ima ,wi ,he, cls, xmin ,ymin ,w ,h])
    column_name = ['image_id', 'width', 'height','source', 'x','y','w','h']
    df = pd.DataFrame(annos, columns=column_name)        

    return df

if __name__ == "__main__" :

    # Argument Parser
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, help="txt path")
    ap.add_argument("-img", "--image", required=True, help="images path")
    ap.add_argument("-o", "--output", required=True, help="output csv path ")
    args = vars(ap.parse_args()) 

    # Define class number according to the  classes in the .txt file
    dict = {'0' : 'molec',
            '1': "arrow",
            '2': "text",
            '3': "plus"
                        }      
    # Assign paths        
    ann_path = args["input"]
    img_path = args["image"]
    csv_path = args["output"]  

    data=write_to_csv(ann_path ,img_path  ,dict)      
    # print()
    data.to_csv(csv_path, index=None)
    print('Successfully converted txt to csv. And your output file is {}'.format(args["output"])) 