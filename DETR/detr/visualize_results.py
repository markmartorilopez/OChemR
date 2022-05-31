from util.plot_utils import plot_logs
import cv2
from pathlib import Path

outDir = 'output/'

log_directory = [Path(outDir)]

name = 'loss_mAP'
fields_of_interest = (
    'loss',
    'mAP',
    )

plot_logs(log_directory,
          fields_of_interest,name)

fields_of_interest = (
    'loss_ce',
    'loss_bbox',
    'loss_giou',
    )
name = 'loss_ce_bbox_giou'

plot_logs(log_directory,
          fields_of_interest,name)

fields_of_interest = (
    'class_error',
    'cardinality_error_unscaled',
    )
name = 'class_error_carderrunscaled'

plot_logs(log_directory,
          fields_of_interest,name)   