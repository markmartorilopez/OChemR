# Copyright (c) 2022 rxn4chemistry - Mark Martori Lopez

import numpy as np
class Bbox:                                                             # Class to manipulate Bounding Boxes
    """
    Class to manipulate Bounding Boxes.
    """
    def __init__(self, bbox):
        self.bbox = bbox
        self.x = self.bbox[0]
        self.y = self.bbox[1]
        self.fx = self.bbox[2]
        self.fy = self.bbox[3]

    def centeredPoint(self):                                            # x,y coords of middle of bbox.
        """
        Compute equidistant center point between 2 points.
        """
        centeredPoint = (int((self.x + self.fx) / 2), int((self.y + self.fy) / 2))
        return centeredPoint

    def midPoints(self):                                                # Midpoints = half distant points between consecutive corners.
        """
        Compute center point between all sides of bounding box, check drawing below:
        """
        N = np.array([self.x + int((self.fx - self.x) / 2), self.y])    #  ___.___
        S = np.array([self.x + int((self.fx - self.x) / 2), self.fy])   # |       |
        E = np.array([self.fx, self.y + int((self.fy - self.y) / 2)])   # .       .
        W = np.array([self.x, self.y + int((self.fy - self.y) / 2)])    # |___.___|
        return N,E,S,W

    def mindist(self,pointa,midPoints):                                 # Minimum distance bw point and list of midpoints.
        if len(pointa) == 2 and len(midPoints) == 4:
            dist = []
            for midpoint in midPoints:
                distancia = np.linalg.norm(pointa - midpoint)
                # print(pointa, midpoint, distancia)
                dist.append(distancia)
            return min(dist)
        else:
            print(type(pointa))
            return None

def groupMolecules(bbox_dict,SMILES_dict, labels):
    """
    Checks if symbol "+" in image.
    If true, it joins the 2 molecules that are summing into a single "Molecule" representation.
    """
    try:
        idx = list(SMILES_dict.values()).index(".")                     # Check if "+" (represented as ".") in SMILES_dict        
    except: return bbox_dict, SMILES_dict, labels
    symbolbbox = Bbox(bbox_dict[idx])                                   # if found, get its coords.
    centeredPoint = symbolbbox.centeredPoint()                          # Get centered point of + symbol.
    mindistances = {}
    threshold = 20                                                      # Molecules further than 20pxls not taken into account.
    for it,bbox in enumerate(list(bbox_dict.values())):                 # check each bbox
        if labels[it] != 0:                                             # if its a molecule
            continue
        else:
            currentbbox = Bbox(bbox)                                    # Bbox class
            mindist = currentbbox.mindist(centeredPoint,currentbbox.midPoints()) # Get distance of closest molecules to "+"
            if mindist <= threshold:                                    # Store its distance if closer than threshold.
                mindistances[it] = mindist

    if len(mindistances.keys()) < 2:
        print(f"Symbol + found, but not molecules close enough detected. May be that Obj. detector did not find the molecules.")
        return bbox_dict, SMILES_dict, labels

    closest = min(mindistances, key=mindistances.get)                   # Get closest molecule
    mol1bbox = bbox_dict[closest]                                       # Get bbox of such molecule
    newmol = SMILES_dict[closest]                                       # Initiate new molecule representation

    del bbox_dict[closest]                                              # Modify dictionaries
    del mindistances[closest]
    del SMILES_dict[closest]

    sec_closest = min(mindistances, key=mindistances.get)               # Get 2nd closest molecule
    mol2bbox = bbox_dict[sec_closest]                                   # Get bbox of 2nd closest mol.
    newmol = newmol+"."+SMILES_dict[sec_closest]                        # Build new joined_molecules

    joined_bbox = [min(mol1bbox[0], mol2bbox[0]), min(mol1bbox[1], mol2bbox[1]), max(mol1bbox[2], mol2bbox[2]),
                            max(mol1bbox[3], mol2bbox[3])]              # Join 2 bboxes into 1.

    print(f"Molecule joined placed in pos {sec_closest} in dict.")      # Modify dictionaries:
    bbox_dict[sec_closest] = joined_bbox                                # 2nd mol detected -> new Joined Mol.
    SMILES_dict[sec_closest] = newmol                                   # and first mol detected position will be deleted.

    return bbox_dict, SMILES_dict, labels                               # Return modified dicts.


def splitMolecfromText(mindistances, otherpoint, bbox_dict,threshold):
    """
    Gets all mindistances if there are +1 molecule very close to arrow edge.

    And decides wether the molecules is the one pointing to arrow or it is just text above the arrow.
    Ex:
        #      1 - If there is a molecule in the arrow process [mol2], it may be closer to start/end points than the (mol1 or mol3)
        #                          mol2 - text
        #           Ex: molec1 sp--------->ep molec3
        #
        # Solution:
        #                          mol2 - text
        #      1 -  Ex: molec1 sp--------->ep molec3
        #      if mindistance(sp,mol2) < mindistance(sp,molec1), we compute mindistance(ep,molec1) and...
        #      
        #       ...mindistance(ep,mol2) and we assign the max distant molec (molec1) as closest to sp, and mol2 as "text"
    """
    mindistances_w_otherpoint = {}                                      # New dict                
    for k in mindistances:                                              # Iterate through min distances between one edge of arrow and molecules.
        if mindistances[k] < threshold:                                 # If close enough to be considered a relevant molecule for the arrow:
            bbx = Bbox(bbox_dict[k])
            # Get distance with other arrow point
            mindistances_w_otherpoint[k] = bbx.mindist(otherpoint,bbx.midPoints()) # Get distance with other edge of arrow.
    
    opposite_further = max(mindistances_w_otherpoint, key=mindistances_w_otherpoint.get) # Get index of max distance.
    del mindistances_w_otherpoint[opposite_further]                     
    mid_closest = list(mindistances_w_otherpoint.keys())
    return opposite_further, mid_closest                                # Return index of molecule that is closer to current edge, but further to opposite edge of arrow.


def findClosestType(sp,ep, bbox_dict,labels, arrowidx, typeobj = 0, threshold = 30, debugging = False):
    """                     mol_middle or text
    [mol_before] :-∞-∞- sp---current_arrowidx--->ep -∞-∞-: [mol_after]

    Finds closest typeobjects [molecules or text] to current arrow.

    INPUTS:
        sp : list -> startpoint of arrow
        ep : list -> endpoint of arrow
        bbox_dict : dict -> dict containing all bboxes of image.
        labels : list -> labels ordered as in bbox_dict.
        arrowidx : int -> index of current arrow.
        threshold : int -> limit of distance explored to find objects close.
    OUTPUTS:
        prev_closest : int -> index of MOLECULE from which the arrow starts. if typeobj = 2, then set to none.
        mid_closest  : int -> index of objects found in the middle of the arrow / process.
        post_closest : int -> index of MOLECULE to which the arrow points. if typeobj = 2, then set to none.
    """
    prev_mindistances = {}
    post_mindistances = {}
    # Get start and end arrow points:
    sp = [sp[0] + bbox_dict[arrowidx][0], sp[1] + bbox_dict[arrowidx][1]] # get startpoint arrow
    ep = [ep[0] + bbox_dict[arrowidx][0], ep[1] + bbox_dict[arrowidx][1]] # get endpoint arrow
    prev_num_closest_obj = 0                                            # Num of objs. closer than thresh to startp.
    post_num_closest_obj = 0                                            # Num of objs. closer than thresh to endp.

    for it,key in enumerate(bbox_dict):                                 # For each bbox.
        if labels[key] == typeobj:                                      # Check if we are interested on images (0) or text (2)
            currentbbox = Bbox(bbox_dict[key])

            # From start point of arrow:
            mindist = currentbbox.mindist(sp,currentbbox.midPoints())   # Get minimum distance, from sp to all 4 midpoints of bbox checked.
            prev_mindistances[it] = mindist if mindist < threshold else threshold  # if the distance is longer than threshold not interesting.
            if mindist < threshold: prev_num_closest_obj += 1           # Store num of objects considered to be closer to start point.

            # From end point of arrow:
            mindist = currentbbox.mindist(ep,currentbbox.midPoints())   # Get minimum distance, from ep to all 4 midpoints of bbox.
            post_mindistances[it] = mindist if mindist < threshold else threshold
            if mindist < threshold: post_num_closest_obj += 1

    if debugging:
        print(f"\nPREV= {prev_mindistances}")
        print(f" POST = {post_mindistances}\n")
    mid_closest = []                                                    # store objects above arrows (text or molecules in the process of the arrow but not in the edges).
    if typeobj == 0:                                                    # if intereted on Molecules:
        if prev_num_closest_obj < 2:                                    # if there is only 1 molecule close to startpoint
            prev_closest = min(prev_mindistances, key=prev_mindistances.get)

        else: # What if there is a second or third very close molecule in the middle?
            post_further, mid_closest = splitMolecfromText(prev_mindistances,ep, bbox_dict,threshold)
            prev_closest = post_further                                 # Call function to decide whether each molecule should be considered text or not.

        if post_num_closest_obj < 2:                                    # if there is only 1 molecule close to endpoint of arrow
            post_closest = min(post_mindistances, key=post_mindistances.get)

        else: # What if there is a second or third very close molecule in the middle?
            prev_further, mid_closest1 = splitMolecfromText(post_mindistances,sp, bbox_dict,threshold)
            post_closest = prev_further                                 # Call function to decide whether each molecule should be considered text or not.

            mid_closest = mid_closest + list(set(mid_closest) - set(mid_closest1))
            # If mol detected as "above molecule" is also a pointing arrow, do not take it.
            mid_closest = [mol for mol in mid_closest if (str(mol) != str(prev_closest) and str(mol) != str(post_closest))]
            mid_closest = set(mid_closest)

    else:                                                               # if we process text, we want all text closer than threshold
        prev_closest = None
        post_closest = None
        for k in prev_mindistances:
            if prev_mindistances[k] < threshold:
                mid_closest.append(k)
        for k in post_mindistances:
            if post_mindistances[k] < threshold:
                mid_closest.append(k)

        mid_closest = set(mid_closest)
    
    return [prev_closest,mid_closest,post_closest]


def orderArrows(unorderedReaction, debugging = False):
    """
    Restructure dictionary to follow reactions direction.
    Input: unorderedReaction : dict
            {arrow2: {prevmol, text, postmol}, arrow1: {...} }

    Output: orderedReaction : dict
            {{arrow1: {prevmol, text, postmol}, arrow2: {...} }
    """

    orderedReaction = {}
    SMILESreaction = ""
    prevmols = []
    postmols = []
    for arrow in unorderedReaction.keys():
        prevmols.append([unorderedReaction[arrow]["prev_mol"],arrow])
        postmols.append([unorderedReaction[arrow]["post_mol"],arrow])

    # Find 1st Molecule
    notfound = False
    for prevmol in prevmols:
        notfound = prevmol[0] not in (item for sublist in postmols for item in sublist)
        if notfound:
            firstmol = prevmol[0]
            orderedReaction[prevmol[1]] = unorderedReaction[prevmol[1]] 
            SMILESreaction += prevmol[0]+ ">>"
    if len(orderedReaction.keys()) != 1:
        print(f"Arrows start/end points seem to be wrongly chosen. Do not take this results into consideration.")
        return False,False
        
    # Join the rest one by one on the queue:
    prevmol = firstmol
    for arr_it in range(0,len(unorderedReaction.keys())):
        for arrow in unorderedReaction:
            if unorderedReaction[arrow]["prev_mol"] == prevmol:
                postmol = unorderedReaction[arrow]["post_mol"]
                orderedReaction[arrow] = unorderedReaction[arrow]
                SMILESreaction += postmol+">>"
                prevmol = postmol

    if debugging:
        print(f"Ordered = {orderedReaction}")
        print(f"SMILES reaction = \n {SMILESreaction}")


    return orderedReaction, SMILESreaction
    
#
# ---------------   DATA TO PLAY WITH ABOVE FUNCTIONS -------------
#
# labels = [1, 2, 2, 0, 2, 0, 0, 2, 0, 1, 2, 2, 2, 0, 1, 0, 2, 1, 2, 1, 2, 2, 2, 2]
# SMILES_dict = {
# 0: [(16, 106), (77, 20)], 
# 1: [''], 
# 2: [''], 
# 3: 'C=C1CC2(C#N)C3(C)CC1C1CCNC12CC1=C3Cc2ccc(C)c(-c3c([O-])ccc4[nH]c5c(c34)CC3C[C@H]4C5(C)CC(C)([C@@H](C)CCO)C34CC3CC3)c21', 4: [''], 5: 'molecule5', 6: 'C=Cc1ccc2[nH]cc(C[C@H](N)C(=O)OF)c2c1', 7: [''], 8: 'molecule8', 9: [(151, 14), (4, 14)], 10: ['--:-nun-FoUr5--o'], 11: ['-à--------nrMHCFIF'], 12: ['-)-LL-I.CoOK-co-work-.org.-Chem.:'], 13: 'BCc1ccc2[nH]c3c(c2c1-c1c(O)ccc(C)c1C1=C(C)C2CC4/C(=C\\C)CN2C(C1)C4CC)CC1C(CC)C24CC3N1[C@@H]2/C4=C/C.C.CC=O.CC=O', 14: [(60, 9), (114, 98)], 15: 'C/C=C(\\C)CN1C2CCC(=O)[C@]1(C)Cc1c2[nH]c2ccc(C)cc12.O', 16: [''], 17: [(4, 14), (154, 14)], 18: [''], 19: [(9, 12), (53, 13)], 20: [''], 21: ['-:n-'], 22: [''], 23: ['-----DDI3,2,-78°oCtorV112912,-viel,YIIYIXX-4--thenMeOH,r.t,20']}


# bbox_dict ={
#     0: [75, 202, 156, 319], 
#     1: [356, 107, 386, 123], 2: [0, 6, 23, 21], 3: [412, 309, 596, 521], 4: [457, 510, 558, 525], 5: [153, 156, 311, 314], 6: [3, 10, 145, 99], 7: [187, 26, 227, 40], 8: [496, 156, 639, 280], 9: [341, 209, 486, 224], 10: [465, 11, 569, 46], 11: [351, 177, 483, 205], 12: [138, 61, 287, 104], 13: [6, 306, 193, 517], 14: [466, 47, 573, 149], 15: [285, 0, 446, 106], 16: [494, 72, 535, 89], 17: [225, 409, 384, 425], 18: [284, 430, 309, 443], 19: [184, 43, 236, 57], 20: [404, 233, 431, 246], 21: [247, 292, 403, 316], 22: [46, 255, 72, 268], 23: [226, 371, 373, 409]} # new molecule above arrow fast.


# for key in bbox_dict: 
#     if labels[key] == 1: # Iterate dict keys in case we deleted a molecule index when joining molecules that are summing.
#         sp,ep = SMILES_dict[key]
#         molidxs = findClosestType(sp,ep, bbox_dict, labels,key, typeobj = 2, threshold = 45)