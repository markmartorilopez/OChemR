with open('labels.txt','r') as f:
        for line in f.readlines():
            obj = line.strip().split()
            # x = int(float(obj[1]) * wi)
            # y = int(float(obj[2]) * he)
            # w = int(float(obj[3]) * wi)
            # h = int(float(obj[4]) * he)
            size = (wi,he)
            bbox = [float(obj[1]),float(obj[2]),float(obj[3]),float(obj[4])]

            b = rescale_bboxes(bbox,size)
            
            cv2.rectangle(img,(int(b[0]), int(b[1])),(int(b[2]),int(b[3])), color=(255,0,100), thickness = 2)
            cv2.imwrite('2image_labelled.png',img)