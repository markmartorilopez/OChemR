# Copyright (c) 2022 rxn4chemistry - Mark Martori Lopez
# Data Load
def prepareMolecules(path):
    """
    Get molecules in a .csv file with several attributes. Get only the name and its SMILES forms.
    Return a list of molecules with the SMILES form.
    
    I:
        path: str
            path where the .csv file with molecules is found.
        
    O:
        molecules_SMILES : list
            list of molecules in SMILES format.
        
        molecules_names ; list
            list of molecules names. In same order as molecules_SMILES

    molecules_SMILES = []
    with open(path) as infile:
        for it,line in enumerate(infile.readlines()):
            if it > 7:
                molecules_SMILES.append(line.strip().split()[1][0:-1])
    # return
    return molecules_SMILES
    """    
    molecules_SMILES = []
    with open(path) as infile:
        for line in infile.readlines():
            molecules_SMILES.append(line)
    # return
    return molecules_SMILES