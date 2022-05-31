#!/usr/bin/env bash
#BSUB -q x86_24h
#BSUB -n 1
#BSUB -gpu "num=2"
#BSUB -M 32876
#BSUB -e "err/DETR.sterr.%J"
#BSUB -o "logs/DETR.stdout.%J"
#BSUB -J "GPUscocoDETR"

### source /etc/profile.d/modules.sh

### ml use /opt/share/modulefiles/x86_64/
### ml conda/miniconda3/4.9.2
### conda activate detectron

### From Scratch:
## python main.py --dataset_file "arrow" --arrow_path "/dccstor/arrow_backup/images/" --epochs "300"

### From own pre-trained weights:
torchrun --nproc_per_node=2 main.py --dataset_file "arrow" --arrow_path "/dccstor/arrow_backup/images/" --epochs "30" --resume "/dccstor/arrow_backup/weights/detr-r50-e632da11.pth"

### From COCO or DocSegTr pre-trained weights:
# python main.py --dataset_file "arrow" --arrow_path "/dccstor/arrow_backup/images/" --epochs "30" --output_dir "output_longerepochs/" --resume "output/loss1_7.pth"

# python main.py --dataset_file "arrow" --arrow_path "/dccstor/arrow_backup/images/" --epochs "300" --resume "output_COCO/checkpoint.pth"


### LETS TRY WITH THIS ONE 
### python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --arrow_path "/dccstor/arrow_backup/images/" --lr "1e-5" --epochs "300" --resume "output_COCO/checkpoint.pth"



###### FINE-TUNE - From COCO pre-trained weights:
# output
# python main.py --dataset_file "arrow" --arrow_path "/dccstor/arrow_backup/images/" --epochs "50" --output_dir "output/" --resume "/dccstor/arrow_backup/weights/detr-r50-e632da11.pth" --lr "1e-3" --lr_backbone "1e-4" --lr_drop "42" --weight_decay "1e-2" --clip_max_norm "0.15" --dropout "0.1"

# output1
# python main.py --dataset_file "arrow" --arrow_path "/dccstor/arrow_backup/images/" --epochs "15" --output_dir "output1/" --resume "/dccstor/arrow_backup/weights/detr-r50-e632da11.pth" --lr "1e-5" --lr_backbone "1e-5" --lr_drop "7" --weight_decay "1e-5" --clip_max_norm "0.05" --dropout "0.2"

# output2
# python main.py --dataset_file "arrow" --arrow_path "/dccstor/arrow_backup/images/" --epochs "15" --output_dir "output2/" --resume "/dccstor/arrow_backup/weights/detr-r50-e632da11.pth" --lr "1e-2" --lr_backbone "1e-3" --lr_drop "7" --weight_decay "1e-3" --clip_max_norm "0.1" --dropout "0"

#output3
# python main.py --dataset_file "arrow" --arrow_path "/dccstor/arrow_backup/images/" --epochs "15" --output_dir "output3/" --resume "/dccstor/arrow_backup/weights/detr-r50-e632da11.pth" --lr "1e-4" --lr_backbone "1e-5" --lr_drop "7" --weight_decay "1e-5" --clip_max_norm "0.1" --dropout "0.1"


# detr-r50-e632da11.pth