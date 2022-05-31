import os
import cv2

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import numpy
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from d2.detr import  add_detr_config

import torch
torch.cuda.empty_cache()

cfg = get_cfg()
add_detr_config(cfg)
cfg.merge_from_file("/dccstor/arrow_backup/output_detr2/config.yaml")
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR,"best_weights.pth") # '16loss_weights.pth')#
cfg.DATASETS.TEST = ("my_dataset_test", )
cfg.MODEL.DETR.NUM_CLASSES = 6
cfg.MODEL.DETR.NUM_OBJECT_QUERIES = 100
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set the testing threshold for this model

predictor = DefaultPredictor(cfg)
test_metadata = MetadataCatalog.get("my_dataset_test")

import glob
threshold = 0.39
tested_images = ['../data/test/*png','../../externalDataset/*png']
print(f"Threshold = {threshold}\n")
for paths in tested_images:
    if 'test' in paths: # test
        stripped = '../data/test/'
    else:
        stripped = '../../externalDataset/'
    for imageName in glob.glob(paths): # data/arrow_dt/test/*png
        print(f"Image {imageName.strip(stripped)}:")
        im = cv2.imread(imageName)
        outputs = predictor(im)
        keep = outputs['instances'][outputs['instances'].scores > threshold]
        scores_keep = keep.scores
        print(f"Scores = {scores_keep}")
        v = Visualizer(im[:, :, ::-1],
                        metadata=test_metadata, 
                        scale=0.8
                        )

        # out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        out = v.draw_instance_predictions(keep.to("cpu"))
        # cv2_imshow(out.get_image()[:, :, ::-1])
        filename = imageName.strip(stripped) #'data/arrow_dt/test/')
        cv2.imwrite('test_results/'+filename, out.get_image()[:, :, ::-1])
        print(f"Test image done : {filename}")
