"""
Example Test:
python test_image.py \
    --images-dir "PATH_TO_IMAGES_DIR" \
    --result-dir "PATH_TO_RESULT_DIR" \
    --pretrained-weight ./pretrained/SGHM-ResNet50.pth

Example Evaluation:
python test_image.py \
    --images-dir "PATH_TO_IMAGES_DIR" \
    --gt-dir "PATH_TO_GT_ALPHA_DIR" \
    --result-dir "PATH_TO_RESULT_DIR" \
    --pretrained-weight ./pretrained/SGHM-ResNet50.pth

"""

import argparse
import os
import glob
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision.utils import save_image

from model.model import HumanSegment, HumanMatting
import utils
import inference

# --------------- Arguments ---------------
parser = argparse.ArgumentParser(description='Test Images')
parser.add_argument('--images-dir', type=str, required=True)
parser.add_argument('--result-dir', type=str, required=True)
parser.add_argument('--gt-dir', type=str, default=None)
parser.add_argument('--pretrained-weight', type=str, required=True)

args = parser.parse_args()

if not os.path.exists(args.pretrained_weight):
    print('Cannot find the pretrained model: {0}'.format(args.pretrained_weight))
    exit()

# --------------- Main ---------------
# Load Model
model = HumanMatting(backbone='resnet50')
model = nn.DataParallel(model).cuda().eval()
model.load_state_dict(torch.load(args.pretrained_weight))
print("Load checkpoint successfully ...")


# Load Images
image_list = sorted([*glob.glob(os.path.join(args.images_dir, '**', '*.jpg'), recursive=True),
                    *glob.glob(os.path.join(args.images_dir, '**', '*.png'), recursive=True)])

if args.gt_dir is not None:
    gt_list = sorted([*glob.glob(os.path.join(args.gt_dir, '**', '*.jpg'), recursive=True),
                    *glob.glob(os.path.join(args.gt_dir, '**', '*.png'), recursive=True)])

num_image = len(image_list)
print("Find ", num_image, " images")

metric_mad = utils.MetricMAD()
metric_mse = utils.MetricMSE()
metric_grad = utils.MetricGRAD()
metric_conn = utils.MetricCONN()
metric_iou = utils.MetricIOU()

mean_mad = 0.0
mean_mse = 0.0
mean_grad = 0.0
mean_conn = 0.0
mean_iou = 0.0

# Process 
for i in range(num_image):
    image_path = image_list[i]
    image_name = image_path[image_path.rfind('/')+1:image_path.rfind('.')]
    print(i, '/', num_image, image_name)

    with Image.open(image_path) as img:
        img = img.convert("RGB")

    if args.gt_dir is not None:
        gt_path = gt_list[i]
        gt_name = gt_path[gt_path.rfind('/')+1:gt_path.rfind('.')]
        assert image_name == gt_name
        with Image.open(gt_path) as gt_alpha:
            gt_alpha = gt_alpha.convert("L")
        gt_alpha = np.array(gt_alpha) / 255.0

    # inference
    pred_alpha, pred_mask = inference.single_inference(model, img)

    # evaluation
    if args.gt_dir is not None:
        batch_mad = metric_mad(pred_alpha, gt_alpha)
        batch_mse = metric_mse(pred_alpha, gt_alpha)
        batch_grad = metric_grad(pred_alpha, gt_alpha)
        batch_conn = metric_conn(pred_alpha, gt_alpha)
        batch_iou = metric_iou(pred_alpha, gt_alpha)
        print(" mad ", batch_mad, " mse ", batch_mse, " grad ", batch_grad, " conn ", batch_conn, " iou ", batch_iou)

        mean_mad += batch_mad
        mean_mse += batch_mse
        mean_grad += batch_grad
        mean_conn += batch_conn
        mean_iou += batch_iou

    # save results
    output_dir = args.result_dir + image_path[len(args.images_dir):image_path.rfind('/')]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    save_path = output_dir + '/' + image_name + '.png'
    Image.fromarray(((pred_alpha * 255).astype('uint8')), mode='L').save(save_path)

print("Total mean mad ", mean_mad/num_image, " mean mse ", mean_mse/num_image, " mean grad ", \
    mean_grad/num_image, " mean conn ", mean_conn/num_image, " mean iou ", mean_iou/num_image)
