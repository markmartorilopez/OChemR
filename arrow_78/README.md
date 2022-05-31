# OChemR project

## Generate Training data set
### Target Structure:
- images/
    - json_annotation.json  # JSONs for DETR training.
    - json_annotval.json    #
    - json_annotest.json    #
    - train.csv             # Pixel Mean and Pixel St.Dev.
    - train/
    - val/
    - test/
    - labelled_train/
    
##### Step 1: Set proper paths:
 - folder: arrow_78
 - script : params.json
        - train_path = "location where train images will be" ex: images/train/
        - labelled_path = "location where train images will be" ex: images/train/



##### Step 2: Set proper params: - generate 60k images. Check: info_params.py for more inf.
 - folder: arrowDS/params/
 - script: params.json
        -   {
	            "dataset_params": 
                     {   
                         "train_path" : "images/train/",
                         "labelled_path" : "images/labelled/",
                         "img_width" : [512,650,800,1024],
			    "img_height" : [512,650,800,1024],
			    "molecules_sizes" : [3,6,8,10],
			    "molecules_rotations" : [0,30,100,330],
			    "num_molecules_per_reaction" : 12,
			    "num_reactions_per_epoch" : 15000,
			    "epochs" : 4}
}

##### Step 3: Run.
 - folder: arrow_78
        - terminal: bash < arrow.lsf # will run main.py with parameters.

##### Step 4: Check progress.
 - folder: arrow_78/logs/currentDatefolder/
        - code experiment_log.txt

##### Step 5: Loop for validation images.
 - Remember to change "train_path" in all scripts to validation path. ex: images/val/

##### Step 6: Create .json files to run DETR.
 - folder: DatasetsConversion/_Yolo2COCO/
 - script: main.py  # set proper categories
        - classes = [
                "molec",
                "arrow",
                "text",
                "plus"
            ]

 - TRAIN:
 - script: run.lsf
        - --path location where train images are. ex: images/train/
        - --output json_annotation.json
 - terminal: bash < run.lsf

 - VAL:
 - script: runval.lsf
        - --path location where validation images are. ex: images/val/
        - --output json_annotval.json
 - terminal: bash < runval.lsf

 - TEST:
 - Add images to test folder if not done yet. ex: images/test/
 - script: runtest.lsf
        - --path location where validation images are. ex: images/test/
        - --output json_annotest.json
 - terminal: bash < runtest.lsf

##### Step 7: Move json file to images/ folder:
 - folder: DatasetConversion/_Yolo2COCO/output/
        - terminal: mv json* "path to images/ fodler" ex: images/

## DETR Custom Dataset Training:
 - ##### Step 0: Clone DETR: 
 - terminal: git clone https://github.com/facebookresearch/detr.git

 - ##### Step 1: Create own arrow.py dataset builder:
 - folder: detr/datasets/
 - terminal: cp coco.py arrow.py  # let's get coco.py as backbone.
 - script: arrow.py
        - Change all coco occurrences to arrow.
        - Set:
            PATHS = {
                "train" : ("images/train" , "json_annotation.json"),
                "val" : ("images/val/" , "json_annotval.json")
            }
 - ##### Step 2: Modify __init__.py to use own dataset builder:
 - folder: detr/datasets/
 - script: __init__.py     Add the following:
        - from .arrow import build as build_arrow  # import own builder.
        - Under line 20 add:
            - if args.dataset_file == 'arrow':
                return build_arrow(image_set, args)

 - ##### Step 3: Edit main.py to use own parameters:
 - folder: detr/
 - script: main.py
        - --num_queries, default = 100  # 72 is the max. #obj in dataset (05/03/22). Leave a margin for test images and bad queries. Set to 100.
        - --coco_path change to --arrow_path

 - ##### Step 4: Edit detr.py to use own num.categories.
 - folder: detr/models/
 - script: detr.py
        - line 305: comment:
            - num_classes = 20 if args.dataset_file != "coco" else 91
        - instead, add:
            - if args.dataset_file == "arrow":
                num_classes = 4 + 1    # categories (within the code later on will sum up 1 for N/A cls.)

 - ##### Step 4.2: Get pixel mean and pixel STDev from Custom Dataset.
 - folder: arrow_78/data/
 - script: metrics_data.ipynb
              - Run all cells and check pixel mean and std.
 
 - Edit arrow.py with proper values:
 - folder: detr/datasets/
 - script: arrow.py
              - def make_arrow_transforms(image_set):
                     normalize = T.Compose([
                            T.ToTensor(),
                            T.Normalize([ PIXEL MEAN ], [ PIXEL STD ])
                     ])
 

 - ##### Step 5.0: Run DETR from scratch or OWN pre-trained weights. Good results w. 200 ep.
 - folder: detr/
 - script: train_baseline_DETR.lsf
        - 1st time:
            - python main.py --dataset_file "arrow" --coco_path "images/" --lr "1e-5" --epochs "300" 
        - Resume training with prev. weights:
            - python main.py --dataset_file "arrow" --coco_path "images/" --lr "1e-5" --epochs "300" --resume "output/checkpoint.pth"

 - TIP: * *  Check details per epoch in logs/stdout.out file;Time x epoch -> by f+search = "eta"  * *

 - ##### Step 5.1: Run DETR - COCO pre-trained weights: Should get good results w. 10 ep.
 - Download model DETR R50 from: https://github.com/facebookresearch/detr  in: detr/
 - Edit main.py to get same number of categories.
 - Before weights are loaded to model, let's delete nontrained parameters:
 - script: main.py
        - Under:
            else:
                checkpoint = torch.load(args.resume, map_location='cpu')

        -   # HERE
        -       # NOT HERE
        -   del checkpoint["model"]["class_embed.weight"]
            del checkpoint["model"]["class_embed.bias"]
            del checkpoint["model"]["query_embed.weight"] 

        - One line below, edit:
            model_without_ddp.load_state_dict(checkpoint['model'])
        - to:
            model_without_ddp.load_state_dict(checkpoint['model'], strict=False)

##### Step 6: Check output weights and create plot file - train vs test loss:
 - folder: detr/output/
 - script: plot_results.py
 - terminal: python plot_results.py
 - image: plot.png

 - weights in: checkpoint.pth

 ##### Step 6.1: Check current training status:
 - folder: "decided logs folder for stdout.out files"

 ##### Step Extra: If Dataset format needed -> csv. (not in this project)
 - folder: DatasetsConversion/COCO2csv/
 - script: main.py
              -     dict = {
                     '0' : 'molec',
                     '1': "arrow",
                     '2': "text",
                     '3': "plus"
                                   } 
 - Create train.csv file, to calculate pixel mean and stdev.
 - terminal: python main.py -i images/train/ -img images/train/ -o train.csv
 - terminal: mv train.csv images/


## DETR Custom Dataset Inference:
##### Step 1: Code test.py file from scratch and run inference.
 - folder: detr/
 - script: attention_DETR.py   # Should be already created in this github. Contact: Mark Martori Lopez if not.
 - terminal: bash < inference_DETR.lsf
 - or
 - python attention_DETR.py

        







 