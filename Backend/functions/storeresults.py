# Copyright (c) 2022 rxn4chemistry - Mark Martori Lopez

import json

def storeResults(d,filename,outputdir):
    # # Json output file:
    outputfile = filename.replace(".png",".json")
    # Txt output file:
    with open(outputdir+outputfile,"w") as f:
        json.dump(d,f)


