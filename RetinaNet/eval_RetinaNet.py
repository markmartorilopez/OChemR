# Copyright (c) 2022 rxn4chemistry - Mark Martori Lopez

import os

# Utilities from Facebook's Detectron2
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
# --------

## Prepare Custom Dataset
from detectron2.data.datasets import register_coco_instances
register_coco_instances("my_dataset_train", {}, "images/json_annotation.json", "images/train")
register_coco_instances("my_dataset_val", {}, "images/json_annoval.json", "images/val")

## Training Config
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("my_dataset_train",)
cfg.DATASETS.TEST = ("my_dataset_val",)
cfg.DATALOADER.NUM_WORKERS = 4
cfg.MODEL.WEIGHTS = "output/model_final.pth" # model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 4
cfg.SOLVER.BASE_LR = 0.001
cfg.SOLVER.GAMMA = 0.05
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4
cfg.MODEL.RETINANET.NUM_CLASSES = 4
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.85
predictor = DefaultPredictor(cfg)
evaluator = COCOEvaluator("my_dataset_val", cfg, False, output_dir="output/")
val_loader = build_detection_test_loader(cfg, "my_dataset_val")

## Create Trainer Model from Facebook's Detectron2
from detectron2.engine import DefaultTrainer
class CocoTrainer(DefaultTrainer): # COCO Dataset trained with model in CFG.
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
                os.makedirs("coco_eval", exist_ok=True)
                output_folder = "coco_eval"
        return COCOEvaluator(dataset_name, cfg, False, output_folder)
# --------

## --- Run Evaluation
trainer = CocoTrainer(cfg)
trainer.resume_or_load(resume=False)
inference_on_dataset(trainer.model, val_loader, evaluator)