"""
Example:
python test_video.py \
    --video "PATH_TO_INPUT_VIDEO" \
    --output-video "PATH_TO_OUTPUT_VIDEO" \
    --pretrained-weight ./pretrained/SGHM-ResNet50.pth

"""

import argparse
import os
import glob
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
import torchvision.transforms as transforms

from model.model import HumanSegment, HumanMatting
import utils
import inference

pil_to_tensor = transforms.Compose(
    [
        transforms.ToTensor()
    ]
)


# --------------- Arguments ---------------
parser = argparse.ArgumentParser(description='Test Video')
parser.add_argument('--video', type=str, required=True)
parser.add_argument('--output-video', type=str, required=True)
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

# Load Video
vc = cv2.VideoCapture(args.video)
if vc.isOpened():
    ret, frame = vc.read()
else:
    ret = False

if not ret:
    print('Failed to read the input video: {0}'.format(args.video))
    exit()

num_frame = vc.get(cv2.CAP_PROP_FRAME_COUNT)
fps = vc.get(cv2.CAP_PROP_FPS)
h, w = frame.shape[:2]

infer_size = 1280
if min(h, w) > infer_size:
    if w >= h:
        rh = infer_size
        rw = int(w / h * infer_size)
    else:
        rw = infer_size
        rh = int(h / w * infer_size)
else:
    rh, rw = h, w

rh = rh - rh % 64
rw = rw - rw % 64

# Create output
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(args.output_video, fourcc, fps, (w, h))


# Background Color 
back_np = np.full(frame.shape, 0)
back_np[:, :, 0] = 120
back_np[:, :, 1] = 255
back_np[:, :, 2] = 155


# Process Video 
with tqdm(range(int(num_frame)))as t:
    for c in t:
        if frame is None:
            print("Frame is empty, process finished ...")
            break
        frame_np = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_np)
        frame_tensor = pil_to_tensor(frame_pil)
        frame_tensor = frame_tensor[None, :, :, :].cuda()

        input_tensor = F.interpolate(frame_tensor, size=(rh, rw), mode='bilinear')
        with torch.no_grad():
            pred = model(input_tensor)

        alpha_pred_os1, alpha_pred_os4, alpha_pred_os8 = pred['alpha_os1'], pred['alpha_os4'], pred['alpha_os8']
        pred_alpha = alpha_pred_os8.clone().detach()
        weight_os4 = utils.get_unknown_tensor_from_pred(pred_alpha, rand_width=30, train_mode=False)
        pred_alpha[weight_os4>0] = alpha_pred_os4[weight_os4>0]
        weight_os1 = utils.get_unknown_tensor_from_pred(pred_alpha, rand_width=15, train_mode=False)
        pred_alpha[weight_os1>0] = alpha_pred_os1[weight_os1>0]

        pred_alpha = pred_alpha.repeat(1, 3, 1, 1)
        pred_alpha = F.interpolate(pred_alpha, size=(h, w), mode='bilinear')
        alpha_np = pred_alpha[0].data.cpu().numpy().transpose(1, 2, 0)      

        comp_np = alpha_np * frame_np + (1 - alpha_np) * back_np
        comp_np = comp_np.astype(np.uint8)
        video_writer.write(cv2.cvtColor(comp_np, cv2.COLOR_RGB2BGR))

        ret, frame = vc.read()                                       
        c += 1