# Copyright (c) 2022 rxn4chemistry - Mark Martori Lopez

# Molecules
import rdkit
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from rdkit.Chem.Draw.MolDrawing import DrawingOptions # B&W colors.
# Image
import cv2
from PIL import Image
from PIL import ImageDraw, ImageFont
import numpy as np
import os
import math
# Text
import random
import secrets
import string
# Own functions
from functions.labellingImage import labellingImage

class Molecule():
    " Instantiate a Molecule image. "
    def __init__(self,
                img_width, 
                img_height,
                bond_size,
                rotation):

        self.img_width = img_width 
        self.img_height = img_height
        self.bond_size = bond_size
        self.rotation = rotation

    def drawI(self, mol, letter, current_width, current_height, step, direction = 'None'):
        '''
        Draw the molecule and configure options for each image.
        
        I: 
            mol : SMILES 
                molecule string in SMILES format
                
            output_name : str
                name of the image saved.
                
        O:
            image saved directly in current folder. Folder depend on size and degree.
    
        '''
    
        # Asserts
        assert isinstance(mol, str), 'mol argument must be a string'

        # Variables
        image_path = 'cache/molecule.png'
        # Transform string molecule into Chem Drawing and apply pre-defined rotation and scaling
        molec = AllChem.MolFromSmiles(mol)
        d = Draw.MolDraw2DCairo(self.img_width,self.img_height)
        # d.drawOptions().addStereoAnnotation = True
        d.drawOptions().useBWAtomPalette() # Set B&W default color - updateAtomPalette({k: (0, 0, 0) for k in DrawingOptions.elemDict.keys()})
        d.drawOptions().fixedBondLength = 20
        d.drawOptions().rotate = 0
        d.FinishDrawing()
        d.DrawMolecule(molec)
        d.WriteDrawingText(image_path)

        # Get B.Box of the Molecule
        coords = []
        coords_mol = labellingImage(image_path,'0', current_width, current_height, step) # class 0 = Molecule
        coords.append(coords_mol) # update main coordinates list.

        # Read molecule and labeled molecule:
        img = Image.open(image_path)

        I1 = ImageDraw.Draw(img)

        myFont = ImageFont.truetype("fonts/arial.ttf",int(self.img_height * 0.08))  #.load_default()

        # Add letter in molecule image:
        xpos = int(self.img_width/2 - 4)
        ypos = int(self.img_height - int(self.img_height * 0.156))

        # Add Text to an image --> (x,y)
        I1.text((xpos, ypos), letter, font=myFont, fill=(0, 0, 0))

        # Add bounding box to labelled image for the letter and return it with its coordinates
        w = xpos + int(xpos * 0.3)
        h = ypos + int(ypos * 0.1)
        
        # Update main coordinates list
        coords.append(['2', xpos - 4  + current_width,ypos+current_height, current_width + w, current_height + h]) # class 3 = Letter
        # print(F"Molecule coords = {coords}")
        return 'molecule', img, coords
        
class Symbol():
    " Instantiate an Arrow image with text. "
    def __init__(self,
                img_width, 
                img_height,
                reaction_width):
        
        self.img_width = img_width
        self.img_height = img_height
        self.reaction_width = reaction_width
        self.probabs = [1,1,1,1,1,2,2,2,2,3,4,4,4,4,4] # 1 = curved arrows. 2 = vertical. 3 = diagonal. 4 = horizontal.
        self.percentage_pad = 0
        self.text_padding = 0
        self.num_characters_text = 0
        self.rotations = [0,90,180,270]

    def insertImageOnWhiteBckg(self,bckground,foreground, current_w, current_h, objtype, width_obj, height_obj = None, xpos = None, ypos = None):
        if objtype == "plus":
            # Where to paste im2:
            xpos = random.randint(int(self.img_width / 2 - 10),int(self.img_width / 2 + 10) )
            ypos = int(self.img_height / 2)
            # Paste the molecule to the image with the rectangle.
            bckground.paste(foreground,(xpos,ypos),foreground) # im2 as 3rd param, to add white background.
            height_obj = width_obj
        else:
            
            # Paste the molecule to the image with the rectangle.
            bckground.paste(foreground,(xpos,ypos),foreground) # im2 as 3rd param, to add white background.

        # Store arrow position
        arrow_x_coord = current_w + xpos - 1
        arrow_y_coord = current_h + ypos - 1
        arrow_xend_coord =  arrow_x_coord + width_obj + 4
        arrow_yend_coord = arrow_y_coord + height_obj + 4

        return bckground, arrow_x_coord, arrow_y_coord,arrow_xend_coord, arrow_yend_coord

    def drawArrowPIL(self,im, ptA, ptB, width=2, color=(0,0,0)):
        """Draw line from ptA to ptB with arrowhead at ptB"""
        # Get drawing context
        draw = ImageDraw.Draw(im)
        # Draw the line without arrows
        draw.line((ptA,ptB), width=width, fill=color)
        sizeTip = int(self.img_width * 0.05)
        # Now work out the arrowhead
        # = it will be a triangle with one vertex at ptB
        # - it will start at 95% of the length of the line
        # - it will extend 8 pixels either side of the line
        x0, y0 = ptA
        x1, y1 = ptB
        # Now we can work out the x,y coordinates of the bottom of the arrowhead triangle
        xb = 0.9*(x1-x0)+x0
        yb = 0.9*(y1-y0)+y0

        # Work out the other two vertices of the triangle
        # Check if line is vertical
        if x0==x1:
            vtx0 = (xb-sizeTip, yb)
            vtx1 = (xb+sizeTip, yb)
        # Check if line is horizontal
        elif y0==y1:
            vtx0 = (xb, yb+sizeTip)
            vtx1 = (xb, yb-sizeTip)
        else:
            alpha = math.atan2(y1-y0,x1-x0)-90*math.pi/180
            a = 8*math.cos(alpha)
            b = 8*math.sin(alpha)
            vtx0 = (xb+a, yb+b)
            vtx1 = (xb-a, yb-b)

        # Now draw the arrowhead triangle
        draw.polygon([vtx0, vtx1, ptB], fill=color)
        return im

    def createTextLines(self,img,xtext,ytext):
        min_, max_ = 6 , 11
        xpostexts = []
        # Prepare Draw 
        I1 = ImageDraw.Draw(img)
        myFont = ImageFont.truetype("fonts/arial.ttf",random.randint(int(self.img_height * 0.058),int(self.img_height * 0.0625)))#.load_default()
        yendtext = ytext
        for it in range(0,random.randint(1,4)):
            txt = ''.join(secrets.choice(string.ascii_letters) for x in range(random.randint(min_,max_)))
            I1.text((xtext,yendtext),txt, font=myFont, fill=(0, 0, 0))
            size_txt = I1.textsize(txt, font=myFont)
            yendtext = yendtext + size_txt[1]
            xpostexts.append(size_txt[0])
        
        xendtext = max(xpostexts) if len(xpostexts) > 1 else xpostexts[0]
        return img, int(xendtext), int(yendtext)

    def addTextInImage(self,img,typeImg, coords, current_w, current_h):
        
        if typeImg == "plus":
            # Write text on the 1st third of image
            x = random.randint(10,int(self.img_width * 0.2))
            y = random.randint(int(self.img_width * 0.1),int(self.img_width * 0.25))
            # how many lines?
            img,xend,yend = self.createTextLines(img,x,y)
            coords.append(['2', current_w + x, current_h + y, current_w +  x + xend, current_h + yend])

        elif typeImg == "horizontal":
            # write text on the 1st third of image and on the 3rd-third of image.
            x = random.randint(10,int(self.img_width * 0.2))
            y = random.randint(int(self.img_width * 0.1),int(self.img_width * 0.2))
            img,xend,yend = self.createTextLines(img,x,y)
            coords.append(['2', current_w + x, current_h + y, current_w +  x + xend, current_h + yend])
            # write text on the 1st third of image and on the 3rd-third of image.
            x = random.randint(2,int(self.img_width * 0.08))
            y = random.randint(int(self.img_height / 2) + 20,int(self.img_height / 2 + int(self.img_height * 0.2)))
            img,xend,yend = self.createTextLines(img,x,y)
            coords.append(['2', current_w + x, current_h + y, current_w +  x + xend, current_h + yend])

        elif typeImg == "vertical" or typeImg == "claudators":
            # write text on the LEFT of image.
            x = random.randint(2,int(self.img_width * 0.08))
            y = random.randint(5,int(self.img_height / 2))
            img,xend,yend = self.createTextLines(img,x,y)
            coords.append(['2', current_w + x, current_h + y, current_w +  x + xend, current_h + yend])

        elif typeImg == "diagonal": 
            # text on leftmiddle and rightmiddle or PASS
            x = random.randint(2,int(self.img_width * 0.08))
            y = random.randint(30,int(self.img_height / 2))
            img,xend,yend = self.createTextLines(img,x,y)
            coords.append(['2', current_w + x, current_h + y, current_w +  x + xend, current_h + yend])
        else: # typeImg == "u or d" -> Corners:
            # add text depending on placetext variable.
            if typeImg == "u": # write up... like in plus images.
                self.addTextInImage(img,"plus", coords, current_w, current_h)
            elif typeImg == "d": # write down.
                x = random.randint(2,int(self.img_width * 0.08))
                y = random.randint(int(self.img_height / 2),int(self.img_height / 2 + int(self.img_height * 0.2)))
                img,xend,yend = self.createTextLines(img,x,y)
                coords.append(['2', current_w + x, current_h + y, current_w +  x + xend, current_h + yend])
            elif typeImg == "r" or typeImg =="l":
                x = int(self.img_width / 2) if typeImg == "r" else 10
                y = random.randint(int(self.img_height / 2),int(self.img_height / 2 + int(self.img_height * 0.2)))
                img,xend,yend = self.createTextLines(img,x,y)
                coords.append(['2', current_w + x, current_h + y, current_w +  x + xend, current_h + yend])
            else:
                print(f"mk - Typeimg = {typeImg}, not able to add text.")

        return img, coords
    
    def drawI(self,p_name, current_w, current_h, direction = 'None'):
        coords = []
        y_pos = int(self.img_height / 2) # half of the image height.
        # x_pos = int(self.img_width / 2)
        typeImg = ""
        arrow_x_coord = 0 # Checker...

        if "plus" in p_name: # Draw PLUS
            typeImg = "plus"
            whichplus = 3 if current_w < 115 else random.randint(0,2)
            length_obj = 10
            if whichplus == 3:
                length_obj = 16
            elif whichplus == 2:
                length_obj = 24
            else:
                length_obj = 30

            # Get white image as background
            img = Image.open('symbols/white.png')
            img = img.resize((self.img_width,self.img_height))
            im2 = Image.open('symbols/plus/plus'+str(whichplus)+'.png')

            img,arrow_x_coord,arrow_y_coord,arrow_xend_coord,arrow_yend_coord =  self.insertImageOnWhiteBckg(img,im2, current_w, current_h, "plus", length_obj)
            
            coords.append(['3', arrow_x_coord, arrow_y_coord, arrow_xend_coord,arrow_yend_coord])

        else: # Corner, Vertical, Diagonal and horizontal arrows.

            # Higher probability of -| or |- than vertical or diagonal arrows.
            probab = random.sample(self.probabs,1)[0]

            if probab == 1: # if curved arrow:
                typeImg = "curved"
                img = Image.open('symbols/white.png')
                img = img.resize((self.img_width,self.img_height))

                if self.reaction_width <= 512:
                    dirsize = "512/"
                elif self.reaction_width <= 650:
                    dirsize = "650/"
                elif self.reaction_width <= 800:
                    dirsize = "800/"
                else:
                    dirsize = "1024/"
                MorL = random.sample(("M/","L/"),1)[0]
                arrowpath = random.choice(os.listdir("symbols/corners/"+MorL+dirsize))
                arrowsize = os.path.basename(arrowpath).strip(".png").split('_')
                xposArr = int(arrowsize[0])
                yposArr = int(arrowsize[1])
                arrow_w = int(arrowsize[2])
                arrow_h = int(arrowsize[3])
                typeImg = arrowsize[4]
                im2 = Image.open("symbols/corners/"+MorL+dirsize+arrowpath)
                im2_h,im2_w = im2.size
                if im2_h > self.img_height or im2_w > self.img_width:
                    im2 = im2.resize((self.img_width,self.img_height))

                print(f"We are at: {arrowpath}:")
                img,arrow_x_coord,arrow_y_coord,arrow_xend_coord,arrow_yend_coord = self.insertImageOnWhiteBckg(img,im2,current_w,current_h,"arrow",arrow_w,arrow_h, xposArr, yposArr)

            elif probab == 2: # Vertical
                typeImg = "vertical"
                choose2 = int(random.sample((9,10,11),1)[0])
                img = Image.open('symbols/line/'+str(choose2)+'.png')
                img = img.resize((self.img_width,self.img_height))

                arrow_x_coord = current_w + int(0.68 * self.img_width) if choose2 == 10 else current_w + int(0.65 * self.img_width)
                arrow_y_coord = current_h + int(0.31 * self.img_height)
                arrow_xend_coord = arrow_x_coord + int(self.img_width * 0.15)
                arrow_yend_coord = arrow_y_coord + int(0.38 * self.img_height)

            elif probab == 3: # Diagonal
                typeImg = "diagonal"
                choose2 = random.sample(self.rotations,1)[0]
                img = Image.open('symbols/line/diagonal.png')
                if choose2:
                    img = img.rotate(choose2, Image.NEAREST, expand = 1)
                
                img = img.resize((self.img_width,self.img_height))

                arrow_x_coord = current_w + int(0.258 * self.img_width)
                arrow_y_coord = current_h + int(0.258 * self.img_width)
                arrow_xend_coord = current_w + self.img_width - int(0.258 * self.img_width)
                arrow_yend_coord = current_h + self.img_height - int(0.258 * self.img_width)

            else: # Horizontal arrows
                typeImg = "horizontal"
                choose = int(random.sample((1,1,1,1,2,2,2,2,3,4),1)[0])
                if choose == 1: # add long horizontal arrows:
                    choose2 = random.randint(1,3)
                    img = Image.open('symbols/line/'+str(choose2)+'.png')
                    img = img.resize((self.img_width,self.img_height))                    
                    arrow_x_coord = current_w + int(0.16 * self.img_width)
                    arrow_y_coord = current_h + y_pos - 8
                    arrow_xend_coord = current_w + self.img_width - int(0.16 * self.img_width)
                    arrow_yend_coord = arrow_y_coord + int(self.img_width * 0.22)
                    
                elif choose == 2: # add short horizontal arrows:
                    choose2 = random.randint(4,5)
                    img = Image.open('symbols/line/'+str(choose2)+'.png')
                    img = img.resize((self.img_width,self.img_height))                    
                    arrow_x_coord = current_w + int(0.28 * self.img_width)
                    arrow_y_coord = current_h + int(0.46 * self.img_height)
                    arrow_xend_coord = arrow_x_coord + int(0.45 * self.img_width)
                    arrow_yend_coord = arrow_y_coord + int(0.10 * self.img_height)
                
                elif choose == 3: # add different horizontal arrows:
                    handmade = random.randint(0,1)
                    if handmade:
                        ptA = (random.randint(int(0.1 * self.img_width), int(0.4 * self.img_width)),y_pos)
                        ptB = (random.randint(int(0.6 * self.img_width), int(0.9 * self.img_width)),y_pos)

                        img = Image.open("symbols/white.png")
                        img = img.resize((self.img_width,self.img_height))
                        widtharr = random.randint(1,2)
                        if direction == "right":
                            img = self.drawArrowPIL(img, ptA, ptB, width=widtharr, color=(0,0,0))
                        else:
                            img = self.drawArrowPIL(img, ptB, ptA, width=widtharr, color=(0,0,0))
                        arrow_x_coord = current_w + ptA[0] - 4
                        arrow_y_coord = current_h + ptA[1] - 4
                        arrow_xend_coord = arrow_x_coord + ptB[0] 
                        arrow_yend_coord = arrow_y_coord + 10

                    else:
                        choose2 = random.randint(6,7)
                        img = Image.open('symbols/line/'+str(choose2)+'.png')
                        img = img.resize((self.img_width,self.img_height))                        
                        arrow_x_coord = current_w + int(0.2 * self.img_width)
                        arrow_y_coord = current_h + int(0.36 * self.img_height)
                        arrow_xend_coord = arrow_x_coord + int(0.68 * self.img_width)
                        arrow_yend_coord = arrow_y_coord + int(0.25 * self.img_height)
                    
                else: # add claudators without bbox.
                    typeImg = "claudators"
                    choose2 = random.randint(1,2)
                    img = Image.open('symbols/line/claudator'+str(choose2)+'.png')
                    img = img.resize((self.img_width,self.img_height))
            
            if arrow_x_coord: # Do not store, claudators [ ] coordinates.
                coords.append(['1', arrow_x_coord, arrow_y_coord, arrow_xend_coord,arrow_yend_coord])

        # --------------- ADD TEXT -----------------
        img, coords = self.addTextInImage(img, typeImg, coords, current_w, current_h)
        print(f"And Type img = {typeImg}")
        return 'arrow',img.convert("RGB"), coords


class WhiteBackground():
    def __init__(self,
                img_w,
                img_h):
        self.img_w = img_w
        self.img_h = img_h
        
    
    def drawI(self,path,current_w, current_h,text):
        white_img = Image.open(path)
        white_img = white_img.resize((self.img_w,self.img_h))
        coords = []
        xpos = 5
        ypos = int(self.img_h / 2) + 20
        # Lets add title or body:
        if random.randint(1,2) == 1: # Title.
            thickness = random.randint(2,3)
            scale = random.randint(3,4)
            if scale < 4:
                num_ch = random.randint(2,4)
            else:
                num_ch = random.randint(3,5)
           
        else: # body
            pass

       # Read molecule and labeled molecule:
        I = ImageDraw.Draw(white_img)
        min_,max_ = 8,12
        myFont = ImageFont.truetype("fonts/arial.ttf",int(self.img_h * 0.08))  #.load_default()
        xtext = random.randint(2,int(self.img_w * 0.2))
        ytext = random.randint(0,50)
        yinit = ytext
        xpostexts = []
        for l in range(2,6):
            txt = ''.join(secrets.choice(string.ascii_letters) for x in range(random.randint(min_,max_)))
            I.text((xtext,ytext),txt, font=myFont, fill=(0, 0, 0))
            size_txt = I.textsize(txt, font=myFont)
            ytext = ytext + size_txt[1]
            xpostexts.append(size_txt[0])        
        coords.append(['2', current_w + xtext, current_h + yinit, current_w + xtext + max(xpostexts),current_h + ytext])
        return 'white_img',white_img.convert("RGB"), coords
        
