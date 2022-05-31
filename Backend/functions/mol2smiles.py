# Copyright (c) 2022 rxn4chemistry - Daniel Probst & Mark Martori Lopez

import rdkit
from rdkit import Chem
from rdkit.Chem import MolToSmiles, MolFromMolBlock
import cv2
import requests

def img_to_smiles(image):
    try:
        cv2.imwrite("tmp.png", image)

        res = requests.post(
            url="https://molvec.ncats.io/molvec",
            data=open("tmp.png", "rb"),
            headers={"Content-Type": "image/png"},
        )

        mol_block = res.json()["molvec"]["molfile"]

        return Chem.MolToSmiles(Chem.MolFromMolBlock(mol_block))
    except:
        return ""

# mol1 = img_to_smiles("../img1.png")
# print(f"Mol1 = {mol1}")
# mol2 = img_to_smiles("../img2.png")
# print(f"Mol2 = {mol2}")