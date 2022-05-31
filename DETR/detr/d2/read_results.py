from curses.textpad import rectangle
import json
import cv2

appeared_imgs = []
relacion = {}
contador = 0
with open('/dccstor/arrow_backup/output_detr/inference/coco_instances_results.json', 'r') as results:
    with open('../data/json_annotest.json','r') as ann:
        annotations = json.load(ann)
        data = json.load(results)
        for it,obj in enumerate(data,1):
            if (it-1) % 100 == 0:
                img_id = obj['image_id']
                img_path = annotations["images"][img_id]["file_name"]
                im = cv2.imread('../data/test/'+img_path)
                print(f"Img_id and path= {img_id, img_path}")
                

                contador = 0

            score = obj['score']
            if score > 0.9:
                contador += 1
                bbox = obj["bbox"]
                cls = obj["category_id"]
                
                # BGR, not RGB.
                if cls == 1:
                    color = (255,0,0) # Blue - Molec
                elif cls == 2:
                    color = (0,255,0) # Green - Arrow
                elif cls == 3:
                    color = (0,0,255) # Red - Text
                elif cls == 4:
                    color = (0,255,255) # Prob yellow? - Letter
                else:
                    color = (0,0,0) # Black - Cbox

                x = bbox[0]
                y = bbox[1]
                sp = (int(x),int(y))
                w = x + bbox[2]
                h = y + bbox[3]
                fp = (int(w), int(h))                
                im = cv2.rectangle(im,sp,fp,color,2)
            
            if it % 100 == 0:
                    cv2.imwrite('test_results/'+img_path,im)
                    print(f"Num of bboxes above threshold = {contador}")

            

                

