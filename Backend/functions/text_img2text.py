# Copyright (c) 2022 rxn4chemistry - Mark Martori Lopez

from doctr.models import ocr_predictor
import json
# Get inputs
from doctr.io import DocumentFile
import glob

def extractText(result):
    """
    This function expects an image containing text and a DocTr model with a det_arch='db_resnet50'.

    It extracts the text from the image.

    Output:
        texts : List
            List of strings found in the image.
    """
    list_of_text = []
    # Max proba post processing rule for selecting the right VIN value among docTR results
    for word in result.pages[0].blocks[0].lines[0].words:
        vin = word.value
        confidence = word.confidence
        list_of_text.append(vin)

    return list_of_text


