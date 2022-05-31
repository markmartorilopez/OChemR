"""
Run inference with DETR and visualize bboxes and encoder-decoder attention.
"""
import os
import cv2
import argparse
from pathlib import Path
from PIL import Image
import random
import numpy as np
import torch
from models import build_model
from datasets.arrow import make_arrow_transforms
import matplotlib.pyplot as plt
import torchvision.transforms as T


import time


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

def filter_boxes(scores, boxes, confidence=0.95, apply_nms=True, iou=0.5):
    keep = scores.max(-1).values > confidence # If one category is scored > threshold
    scores, boxes = scores[keep], boxes[keep] # keep that 'detection'

    # if apply_nms:
    #     top_scores, labels = scores.max(-1)
    #     keep = (boxes, top_scores, labels, iou)
    #     scores, boxes = scores[keep], boxes[keep]
    #     # print(f"Scores after nms = {scores}")
    #     # print(f"Scores after nms = {boxes}")

    return scores, boxes
##    ------------      -----------------------------

def get_images(in_path):
    img_files = []
    for (dirpath, dirnames, filenames) in os.walk(in_path):
        for file in filenames:
            if "Final_reaction" not in file:
                filename, ext = os.path.splitext(file)
                ext = str.lower(ext)
                if ext == '.jpg' or ext == '.jpeg' or ext == '.gif' or ext == '.png' or ext == '.pgm':
                    img_files.append(os.path.join(dirpath, file))

    return img_files

## PLOT function -------------
def plot_one_box(x, img, color=None, label=None, line_thickness=1):
    tl = 2 # line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)

    if label:
        tf = tl #max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 4, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, (45, 230, 230), -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 4, [0, 0, 0], thickness=tf, lineType=cv2.LINE_AA)
## -------------       -------------------------


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=6, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # dataset parameters
    parser.add_argument('--dataset_file', default='arrow')
    parser.add_argument('--data_path', default = "/dccstor/arrow_backup/images/test_small/", type=str)
    parser.add_argument('--data_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='test_results/',
                        help='path where to save the results, empty for no saving')
    parser.add_argument('--device', default='cpu',
                        help='device to use for training / testing')
    parser.add_argument('--resume', default = 'output/checkpoint.pth', help='resume from checkpoint')

    parser.add_argument('--thresh', default=0.95, type=float)

    return parser


CLASSES = [
    'N/A','molec','arrow','text','letter','cbox','plus' 
]

# colors for visualization
COLORS = [[255,255,255],[119, 179, 50], [204, 83, 23], [236, 177, 32],
          [126, 47, 142], [0, 114, 178], [77, 190, 238]]

@torch.no_grad()
def infer(images_path, model, postprocessors, device, output_path):
    model.eval()
    duration = 0

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


        # transform = T.Compose([
        #         T.Resize(800),
        #         T.ToTensor(),
        #         T.Normalize([0.96, 0.96, 0.96], [0.13, 0.13, 0.13])
        #     ])


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
        intime = time.time()
        outputs = model(image)
        outtime = time.time()
        outputs["pred_logits"] = outputs["pred_logits"].cpu()
        outputs["pred_boxes"] = outputs["pred_boxes"].cpu()

        probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]

        bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0,], orig_image.size)
        keep = probas.max(-1).values > threshold
        scores, boxes = filter_boxes(probas,bboxes_scaled,threshold)
        scores = scores.data.numpy()
        boxes = boxes.data.numpy()
        
        
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
        # make these smaller to increase the resolution
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
            # ax_i.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
            #                         fill=False, color='blue', linewidth=3))
            ax_i.axis('off')
            ax_i.set_title(CLASSES[probas[idx].argmax()])
            
        fig.tight_layout()
        plt.savefig(output_path+filename.replace(".png","convf.png"))


        ### PLOT ENCODERS SELF-ATTENTION WEIGHTS:
        f_map = conv_features['0'] # Visualize shape of feature map
        shape = f_map.tensors.shape[-2:]
        im_height,im_width,_ = im.shape

        shape = f_map.tensors.shape[-2:] # get H * W shape of features map of CNN
        sattn = enc_attn_weights[0].reshape(shape + shape) # reshape self-att to a more interpretable way
        H = sattn.shape[-2]
        W = sattn.shape[-1]
        # Lets visualize it:
        # downsampling factor for the CNN, is 32 for DETR
        fact = 32

        # let's select 4 reference points for visualization
        # idxs = [(200, 200), (280, 350), (200, 500), (440, 600)]

        # idxs = [(W / 4, H / 4), (W / 2, H / 2), (W - (W / 4), H // 3), (W - (W / 4), H - (H / 4))]
        idxs = [(W / 4, H / 4), (W / 2, H / 2), (W - (W / 4), H - (H / 4)),(W - (W / 4), H // 3)]

        # here we create the canvas
        fig = plt.figure(constrained_layout=True, figsize=(H * 0.7, 8.5 * 0.7))
        # and we add one plot per reference point
        gs = fig.add_gridspec(2, 4)
        axs = [
            fig.add_subplot(gs[0, 0]),
            fig.add_subplot(gs[1, 0]),
            fig.add_subplot(gs[0, -1]),
            fig.add_subplot(gs[1, -1]),
        ]

        # for each one of the reference points, let's plot the self-attention
        # for that point
        it = 1
        # print(f"Attention = {sattn[..., 200 // fact, 200 // fact]}")
        for idx_o, ax in zip(idxs, axs):
            # idx = (idx_o[0] // fact, idx_o[1] // fact)
            ax.imshow(sattn[..., int(idx_o[1]), int(idx_o[0])], cmap='cividis', interpolation='nearest')
            ax.axis('off') 
            ax.set_title(f'self-attention'+str(it))
            it +=1


        # and now let's add the central image, with the reference points as red circles
        fcenter_ax = fig.add_subplot(gs[:, 1:-1])
        fcenter_ax.imshow(im)
        tim_height,tim_width,_ = im.shape

        idxs = [(tim_width / 4, tim_height / 4), (tim_width / 2, tim_height / 2), (tim_width - (tim_width / 4), tim_height - (tim_height / 4)),(tim_width - (tim_width / 4), tim_height // 3)]

        for it,idx_o in enumerate(idxs,1):
            # scale2 = height / width
            x = idx_o[0]
            y = idx_o[1]
            fcenter_ax.add_patch(plt.Circle((x , y ), fact // 3, color='yellow'))
            fcenter_ax.annotate(str(it), xy=(x , y ), fontsize=16)
            fcenter_ax.axis('off')
        plt.savefig(output_path+filename.replace(".png","sattn.png"))

        infer_time = outtime - intime
        duration += infer_time
        # print("Processing...{} ({:.3f}s)".format(filename, infer_time))
    avg_duration = duration / len(images_path)
    print("Avg. Time: {:.3f}s".format(avg_duration))



if __name__ == "__main__":
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    threshold = args.thresh
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)

    model, _, postprocessors = build_model(args)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
    model.to(device)
    image_paths = get_images(args.data_path)

    infer(image_paths, model, postprocessors, device, args.output_dir)