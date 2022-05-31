# Copyright (c) 2022 rxn4chemistry - Mark Martori Lopez

import os
import sys
import cv2
from PIL import Image
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as T
import time
p = os.path.abspath('../detr/')
sys.path.append(p)
from datasets.arrow import make_arrow_transforms

## Rescaling/Filtering bboxes ----------
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h,
                          img_w, img_h
                          ], dtype=torch.float32)
    return b

def filter_boxes(scores, boxes, confidence=0.9, apply_nms=True, iou=0.5):
    keep = scores.max(-1).values > confidence                           # If one category is scored > threshold
    scores, boxes = scores[keep], boxes[keep]                           # keep that 'detection'
    return scores, boxes
##    ------------      -----------------------------


## PLOT function -------------
def plot_one_box(x, img, color=None, label=None, line_thickness=1):
    tl = 2                                                              # line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)

    if label:
        tf = tl                                                         # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 4, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, (45, 230, 230), -1, cv2.LINE_AA)     # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 4, [0, 0, 0], thickness=tf, lineType=cv2.LINE_AA)
## -------------       -------------------------


CLASSES = [
    'N/A','molec','arrow','text','letter','cbox','plus' 
]

# colors for visualization
COLORS = [[255,255,255],[119, 179, 50], [204, 83, 23], [236, 177, 32],
          [126, 47, 142], [0, 114, 178], [77, 190, 238]]

@torch.no_grad()
def detr_inference(images_path, model, device, output_path, threshold):
    """
    Run inference with selected weights and test images.
    
    INPUT:
        images_path : list
                List of strings belonging to each test image path.

        model : DETRsegm nn.Module
                DETR model.
        
        device : string
                Device to store tensors.
        
        output_path : str
                Path where to store the detections and attention images.

        threshold : float
                IoU threshold between 0 and 1.

    OUTPUT:
        outputs_dict : dict
                Dictionary such that: {img_path : [coordinates of objects, scores]}
    """
    
    model.eval()

    # Dictionary:
    #       img_name : [scores, boxes]
    outputs_dict = {}
    for img_sample in images_path:
        filename = os.path.basename(img_sample)
        print("processing...{}".format(filename))
        orig_image = Image.open(img_sample).convert("RGB")
        width, height = orig_image.size

        transform = make_arrow_transforms("val")
        dummy_target = {
            "size": torch.as_tensor([int(height), int(width)]),
            "orig_size": torch.as_tensor([int(height), int(width)])
        }

        image, targets = transform(orig_image,dummy_target)
        image = image.unsqueeze(0)
        image = image.to(device)


        conv_features, enc_attn_weights, dec_attn_weights = [], [], []
        hooks = [
            model.backbone[-2].register_forward_hook(
                        lambda self, input, output: conv_features.append(output)

            ),
            model.transformer.encoder.layers[-1].self_attn.register_forward_hook(
                        lambda self, input, output: enc_attn_weights.append(output[1])

            ),
            model.transformer.decoder.layers[-1].multihead_attn.register_forward_hook(
                        lambda self, input, output: dec_attn_weights.append(output[1])

            ),

        ]

        outputs = model(image)

        outputs["pred_logits"] = outputs["pred_logits"].cpu()
        outputs["pred_boxes"] = outputs["pred_boxes"].cpu()

        probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]

        bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0,], orig_image.size)
        keep = probas.max(-1).values > threshold
        scores, boxes = filter_boxes(probas,bboxes_scaled,threshold)
        scores = scores.data.numpy()
        boxes = boxes.data.numpy()

        # Create Labels with higher scores
        labels = []
        for i in range(boxes.shape[0]):
            labels.append(scores[i].argmax() - 1)                       # -1 to match DETR classes

        
        ###### MAKE PLOTS OR JUST RETURN OUTPUTS: ----------------------------
        if output_path == "none":
            outputs_dict[img_sample] = [labels, boxes.tolist()]
        #                 ----------------------------
        else:                                                           # If plots required:
            for hook in hooks:
                hook.remove()

            conv_features = conv_features[0]
            enc_attn_weights = enc_attn_weights[0]
            dec_attn_weights = dec_attn_weights[0].cpu()

            # get the feature map shape
            h, w = conv_features['0'].tensors.shape[-2:]

            ### PLOT DETECTED BBOXES
            im = cv2.imread(img_sample)
            for i in range(boxes.shape[0]):
                label = scores[i].argmax()
                confidence = int(scores[i].max() * 100)
                text = f"{label -1} {confidence:.2f}"
                color = COLORS[label]
                plot_one_box(boxes[i], im, color, label=text)

            cv2.imwrite(output_path+filename, im)

            im = cv2.imread(img_sample)
            # Make these smaller to increase the resolution
            dx, dy = 0.05, 0.05

            x = np.arange(0, w, dx)
            y = np.arange(0,h, dy)
            X, Y = np.meshgrid(x, y)
            extent = np.min(x), np.max(x), np.min(y), np.max(y)

            ### PLOT encoder-decoder multi-head attention weights - CONVOLUTIONAL FEATURE MAP OF WHICH OBJECT TO PREDICT:
            fig, axs = plt.subplots(ncols=len(boxes[0:3]), nrows=1, figsize=(18, 6))
            for idx, ax_i, (xmin, ymin, xmax, ymax) in zip(keep.nonzero(), axs, boxes[0:3]):
                ax_i.imshow(dec_attn_weights[0, idx].view(h, w), interpolation='nearest',
                    extent=extent)
                ax_i.axis('off')
                ax_i.set_title(f'query id: {idx.item()}')
                ax_i.imshow(im, alpha=.35, interpolation='bilinear',
                    extent=extent)
                # ----  Add the following code to add rectangle boxes where the model is focusing:
                # ax_i.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                #                         fill=False, color='blue', linewidth=3))
                # ----
            ax_i.axis('off')
            ax_i.set_title(CLASSES[probas[idx].argmax()])

            fig.tight_layout()
            plt.savefig(output_path+filename.replace(".png","convf.png"))

            ### PLOT ENCODERS SELF-ATTENTION WEIGHTS:
            f_map = conv_features['0']                                  # Visualize shape of feature map
            shape = f_map.tensors.shape[-2:]
            im_height,im_width,_ = im.shape
            shape = f_map.tensors.shape[-2:]                            # get H * W shape of features map of CNN
            sattn = enc_attn_weights[0].reshape(shape + shape)          # reshape self-att to a more interpretable way
            H = sattn.shape[-2]
            W = sattn.shape[-1]
            # Lets visualize it:
        
            fact = 32                                                   # From Facebook: downsampling factor for the CNN, is 32 for DETR

            # Let's select 4 reference points for visualization
            idxs = [(W / 4, H / 4), (W / 2, H / 2), (W - (W / 4), H - (H / 4)),(W - (W / 4), H // 3)]

            # Create the 'canvas'
            fig = plt.figure(constrained_layout=True, figsize=(H * 0.7, 8.5 * 0.7))
            # and we add one plot per reference point, plotting its self-attention
            gs = fig.add_gridspec(2, 4)
            axs = [
                fig.add_subplot(gs[0, 0]),
                fig.add_subplot(gs[1, 0]),
                fig.add_subplot(gs[0, -1]),
                fig.add_subplot(gs[1, -1]),
            ]

            it = 1
            for idx_o, ax in zip(idxs, axs):
                ax.imshow(sattn[..., int(idx_o[1]), int(idx_o[0])], cmap='cividis', interpolation='nearest')
                ax.axis('off') 
                ax.set_title(f'self-attention'+str(it))
                it +=1

            # Let's ADD input image in the center of the canvas:
            fcenter_ax = fig.add_subplot(gs[:, 1:-1])
            fcenter_ax.imshow(im)
            tim_height,tim_width,_ = im.shape

            idxs = [(tim_width / 4, tim_height / 4), (tim_width / 2, tim_height / 2), (tim_width - (tim_width / 4), tim_height - (tim_height / 4)),(tim_width - (tim_width / 4), tim_height // 3)]

            for it,idx_o in enumerate(idxs,1):
                x = idx_o[0]
                y = idx_o[1]
                fcenter_ax.add_patch(plt.Circle((x , y ), fact // 3, color='yellow'))   # Add points where to focus.
                fcenter_ax.annotate(str(it), xy=(x , y ), fontsize=16)
                fcenter_ax.axis('off')
            plt.savefig(output_path+filename.replace(".png","sattn.png"))

            outputs_dict[img_sample] = [labels, boxes.tolist()]

    # Return results
    return outputs_dict