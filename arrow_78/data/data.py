# Copyright (c) 2022 rxn4chemistry 
# - Mark Martori Lopez
import rdkit
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from rdkit.Chem.Draw.MolDrawing import DrawingOptions # B&W colors.

def prepareMolecules(path):
    """
    Get molecules in a .txt. Get only its SMILES forms.
    Return a list of molecules with the SMILES form.
    I:
        path: str
            path where the .txt file with molecules is found.
        
    O:
        molecules_SMILES : list
            list of molecules in SMILES format.
    """
    n_mol_used = 0
    with open(path) as infile: # File from which read the molecules
        with open('molecules.txt','w') as outfile: # File to which write them 
            for it,line in enumerate(infile.readlines()):
                if it < 500000:                    
                    newline = line.strip().split()[1]
                    try:
                        molec = AllChem.MolFromSmiles(newline)
                        d = Draw.MolDraw2DCairo(400,400) # If we can draw them, we can use them.
                        d.DrawMolecule(molec)
                        outfile.write(newline+'\n')
                        n_mol_used += 1
                    except:
                        pass              
                else:
                    break
    # return
    print(f"Found and used {n_mol_used} molecules that can be read by rdkit.")
    return 1

a = prepareMolecules('CIDSMILES.txt')
