# Copyright (c) 2022 rxn4chemistry - Mark Martori Lopez

import numpy as np
import cv2
import os

def get_min_radius(img,points):
    """
    Get minimum distance in pixels for both edged points of arrow in image, in order to avoid reaching end of image.
    This radius is used to compute the area when counting #of 0-value pixels to determine end/start point of arrow.
    
    INPUT:
        img : array
            Current Image.
        points : list of tuples
            Start point of arrow, endpoint of arrow: [(x1,y1),(x2,y2)]

    OUTPUT:
        minradius : int
            Smallest number of pixels to reach the end of the image.
    """
    wi,he,_ = img.shape                         
    xdist = []
    ydist = []
    for p in points:
        x,y= p[0],p[1]
        xdist.append(min([abs(wi-x),abs((wi-x) - wi)]))               # Get min distance between point x,y and borders of image.
        ydist.append(min([abs(he-y),abs((he-y) - he)]))

    minradius = 8
    radius = min(min(xdist),min(ydist))                               # Min of height and width.
    if radius < minradius: 
        return radius
    else:
        return minradius

def get_roi(point,img, init_r = 8):
    """
    Get region-of-interest (area of image)
    Using the radius previously calculated in [get_min_radius].

    INPUT:
        point : tuple
            x1, y1 coords of current point of interest.

        img   : array 
            Current img.

        init_r : int -> default = 8
            Minimal radius to reach the end of the image, calculated in get_min_radius.

    OUTPUT: 
        retimg : array
            ROI of image, squared delimited by radius (init_r) used.
        
    """
    for r in range(init_r,0,-1):                                     # Try get ROI from Radius #pxls ... in case of image borders. If for some reason, radius was badly calcualted, let's decrease until finding a valid one.
        retimg = img[
                point[1] - r : point[1] + r, point[0] - r : point[0] + r
                    ]
        if retimg.size != 0:                                         # If ROI is empty, try again with smaller radius.
            break                                                    # else, return ROI
    return retimg

def get_num_consec_pixels(corner,img,imgsave):
    """
    Calcualtes the amount of consecutive 0-value pixels per directions (N.S.W.E) in a corner detected.
    
    INPUT:  
        corner : list
            Current corner.

        img    : array
            Current img.

        imgsave: array
            Image where to save the debugging drawings.

    OUTPUT:
        max(num_consec_pixels) : int
            Maximum consecutive pixels found in any of the 4 directions.
        
        newcorner : tuple
            Move, slightly, the current corner towards the direction with highest consecutive pixels, to allow for bigger and better ROI comparisons later.
    """
    num_consec_pixels = [0,0]                                       # Empty list of min 2 values to perform max operation.
    x = corner[0]
    y = corner[1]
    color1 = (list(np.random.choice(range(256), size=3)))           # Debugging
    color =[int(color1[0]), int(color1[1]), int(color1[2])]         # Debugging
    initpx = 0                                                      # Pixel value to find consecutively.
    d = 500                                                         # distance to check in pxls
    startd = 2 # 6                                                  # Start the search 2 pxls away from corner to avoid starting in a 255-value pixel.
    numlines = 0                                                    # Number of lines found (line = >10 consec pixels.)
    newdir = -1                                                     # Direction where we are checking, None before loop.
    for dir in range(4):                                            # Let's explore in the 4 directions (N.S.E.W)
        numpxls = 0                                                 # num consec pixels, start with 0
        try:                                                        # try - in case we reach the borders of the image.
            for px in range(startd,d):                              # for each pixel in direction we explore
                if dir == 0 :                                       # check right direc
                    move = y, x + px
                elif dir == 1:                                      # check left direc
                    move = y, x - px
                elif dir == 2:                                      # check down direc
                    move = y + px, x
                else:
                    move = y - px, x                                # check up direc

                newpx = img[move]                                   # check value of explored pixel
                
                if np.sum(newpx) == initpx:                         # if all values are 0.
                    cv2.circle(imgsave,(move[1],move[0]),0,color,1) # Draw a dot, to keep track of exploration in debugging.
                    numpxls +=1                                     # Continue exploring
                else:
                    break                                           # Change direction to explore if next pxl is not 0.
        except:
            pass
        if numpxls > 10:                                            # If we find +10 consec pixels...
            newdir = dir                                            # Line is found in that direction so far.
            numlines +=1                                            # One line found.
        if numlines < 2:                                            # If only 1 line in that corner is found...
            num_consec_pixels.append(numpxls)                       # ...we store the amount of pixels.
        else:
            newdir = -1                                             # If there are 2 lines, we may be in a + shape, therefore not an edge of an arrow, so break.
            num_consec_pixels = [0,0]
            break
                                                                    # In order to avoid getting the ROI in an edge, we move 3pxls towards the line.
        if newdir in [0,1]:                 
            newcorner = (corner[0]+2,corner[1]) if newdir == 0 else (corner[0]-2,corner[1])
        elif newdir in [2,3]:
            newcorner = (corner[0],corner[1] + 2) if newdir == 2 else (corner[0],corner[1]-2)
        else:
            newcorner = corner
    return max(num_consec_pixels), newcorner                        # Return max consec pxls found for that corner.

def get_arrow_direction(img,img_path,step, debugging = False):
    """
    Detect start and end point of an image with an arrow, image may contain also text.

    INPUT:  img : image opencv
                Croped image based on bbox found by Obj.Detector (ViT) containing an arrow.
            img_path : str
                Path to main image, to store croped img in case of debugging.
            step    : int
                Arrow id.
    OUTPUT:
            startpoint : list
                [x,y] coords. of start point of arrow. Coords inside cropped image, not overall image.
            endpoint   : list
                [x,y] coords. of end point of arrow.   Coords inside cropped image, not overall image.
    """
    he,wi,_ = img.shape
    startpoint = (0,0)                                              # Prepare returned values
    endpoint = (0,0)
    filename = os.path.basename(img_path)               
    filename = filename.replace(".",str(step)+".")

    # ADD WHITE PADDING TO DETECT EDGES IF ARROW CUTTING WITH BORDERS OF IMG
    img[0:3,0:wi,:] == [255,255,255]
    img[he:he-3,0:wi,:] == [255,255,255]
    img[0:he,0:3,:] == [255,255,255]
    img[0:he,wi:wi-3,:] == [255,255,255]

    # Apply filters to make corners easier to detect:
    image = cv2.GaussianBlur(img, (11,11), 0)                       # Hihglight corners.
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)                   
    corners = cv2.goodFeaturesToTrack(gray,270,0.1,10)              # Detect corners on gray, gaussian image.
    corners = np.int0(corners)

    # Set threshold to facilitate finding lines from corners        # Apply threshold to original image...
    ret,bwimg = cv2.threshold(img,220,254,0) #235                   #...to remove gray values surroinding the arrow.
    ret2,copybwimg = cv2.threshold(img,220,254,0)                   # DEBUGGING

    if len(corners) < 3:                                            # If no corners found or less than 3:
        if len(corners) < 2:
            print(filename+'-> ONLY 1 or none CORNER DETECTED. None arrow was found. ')
            return (0,0),(he,wi)
        
        x1,y1 = corners[0].tolist()[0]                              
        x2,y2 = corners[1].tolist()[0]
        radius = get_min_radius(bwimg,[(x1,y1),(x2,y2)])            # Get radius to be used in get_roi
        first_n_pixels = np.sum(get_roi((x1,y1),bwimg, init_r = radius)) # Check value of roi.
        n_pixels = np.sum(get_roi((x2,y2),bwimg, init_r = radius))  # Check value of roi pixels.

        if n_pixels > first_n_pixels:                               # If higher value, less 0-value (black) pixels...
            endpoint = (x1,y1)                                      # ...and end-arrow has more 0-value pxls ">".
            startpoint = (x2,y2) # LILA
        else:
            endpoint = (x2,y2)
            startpoint = (x1,y1) # LILA
        
        if debugging:
            cv2.circle(copybwimg,startpoint,radius,(255,0,100),1) # LILA
            cv2.circle(copybwimg,endpoint,radius,(100,0,255),1) # ROSA 
            cv2.imwrite("arrows_detected/arr_"+filename,copybwimg)

        return startpoint, endpoint

    all_corners = {}                                                # If +2 corners detected:
    processed_corners = []                                          # Clean corners without consec. 0-val pixels.
    for it,corner in enumerate(corners):                            # For each corner found
        x,y = corner.ravel()
        if gray[y,x] < 230:                                         # If we are not in a very white pxl
            num_consec_pixels,newcorner = get_num_consec_pixels(corner.ravel(),bwimg, copybwimg) # Get num of consec. pxls per that corner.
        else:
            num_consec_pixels = 0
            newcorner = corner
        processed_corners.append(newcorner)                         # Store the new corners, (since we moved 2pxls towards its line)
        all_corners[it] = num_consec_pixels                         # Store num_consec_pixels

    corners_clean = []
    # Extract only cleaned corners:                                 # From all corners, get only the corners with MAX value of num_consec_pixels
    maxv = max(all_corners.values())                                # Get max value.
    pos = list(all_corners.values()).index(maxv)                    # Get index in all_corners of such max value.
    corners_clean.append(processed_corners[pos])                    # Store corner, of such max value index.
    all_corners[pos] = 0                                            # Set it to 0 to get the 2nd max value index.
    
    maxv = max(all_corners.values())                                # Get 2nd max value.
    pos = list(all_corners.values()).index(maxv)                    # Get index of such 2nd max value.
    corners_clean.append(processed_corners[pos])                    # Store corner, of such 2nd max value index.

    # Get Starting/Final pos
    startpoint,endpoint = (0,0),(0,0)

    if len(corners_clean)>1:                                        # If clean corners were found:
        x1,y1 = corners_clean[0]
        x2,y2 = corners_clean[1]

        radius = get_min_radius(bwimg,[(x1,y1),(x2,y2)])            # Same as before: get radius
        first_n_pixels = np.sum(get_roi((x1,y1),bwimg, init_r = radius))
        n_pixels = np.sum(get_roi((x2,y2),bwimg, init_r = radius))  # Compute ROI sum

        if n_pixels > first_n_pixels:                               # Decide which corner belongs to start/end point
            endpoint = (x1,y1)
            startpoint = (x2,y2)
        else:
            endpoint = (x2,y2)
            startpoint = (x1,y1)
        
        if debugging:
            print(f"Filename = {filename}:\n n_p(2ndcorner){n_pixels} <-> fnp(1stcorner){first_n_pixels}.")
            cv2.circle(copybwimg,startpoint,radius,(255,0,100),1) # LILA RBG
            cv2.circle(copybwimg,endpoint,radius,(255,0,255),1) # PINK

    else:
        print(f"Clean corners in arrow {step} can not be detected.")    
    
    if debugging:
        cv2.imwrite("arrows_detected/arr_"+filename,copybwimg)

    return startpoint, endpoint

# for step,fname in enumerate(glob.glob("../arrows/*.png")):        # Run only this script to test.
#     filename = os.path.basename(fname)
#     img = cv2.imread(fname)
#     sp,ep = get_arrow_direction(img,filename,step, True)
