import time

import torch
from torch.backends import cudnn

from backbone import EfficientDetBackbone
import cv2
import numpy as np
import os
import glob
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import folium

from utils.utils import preprocess, invert_affine, postprocess, preprocess_video
from efficientdet.utils import BBoxTransform, ClipBoxes

compound_coef = 0
force_input_size = None  # set None to use default size

# replace this part with your project's anchor config
anchor_ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
anchor_scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]

threshold = 0.2
iou_threshold = 0.1

use_cuda = False
use_float16 = False
cudnn.fastest = True
cudnn.benchmark = True

obj_list = ['BACKGROUND', 'ConcreteCrack', 'Exposure', 'Spalling', 'PaintDamage', 'Efflorescene', 'SteelDefect'] #

# tf bilinear interpolation is different from any other's, just make do
input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536]
input_size = input_sizes[compound_coef] if force_input_size is None else force_input_size


model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list),
                             ratios=anchor_ratios, scales=anchor_scales)
model.load_state_dict(torch.load(f'./efficientdet-d0_14_78240_0.5939819566509892_20230114.pth', map_location=torch.device('cpu')))
model.requires_grad_(False) # efficientdet-d0_14_78240_0.5939819566509892_20230114.pth, efficientdet-d0_26_21924_0.5826629481312854_20221209.pth
model.eval()

if use_cuda:
    model = model.cuda()
if use_float16:
    model = model.half()

def display2(preds, imgs):
    colors = [
        (0, 0, 255),  # red
        (150, 62, 255),  # violetred
        (255, 0, 255),  # magenta
        (250, 206, 135),  # lightskyblue
        (127, 255, 0),  # springgreen
        (0, 165, 255),  # orange
    ]
    for i in range(len(imgs)):
        if len(preds[i]['rois']) == 0:
            continue

        for j in range(len(preds[i]['rois'])):
            (x1, y1, x2, y2) = preds[i]['rois'][j].astype(np.int32)
            cv2.rectangle(imgs[i], (x1, y1), (x2, y2), colors[preds[i]['class_ids'][j]], 2)
            obj = obj_list[preds[i]['class_ids'][j]]
            score = float(preds[i]['scores'][j])

            cv2.putText(imgs[i], '{}, {:.3f}'.format(obj, score),
                        (x1, y1 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        colors[preds[i]['class_ids'][j]], 1)

    return imgs[i]

regressBoxes = BBoxTransform()
clipBoxes = ClipBoxes()

# Video capture
cap = cv2.VideoCapture('/Users/sunghun/Desktop/capstone/video/10.mp4')# '/Users/sunghun/Desktop/capstone/video/5.mp4'
width = int(cap.get(3)) # 가로 길이 가져오기
height = int(cap.get(4)) # 세로 길이 가져오기
fps = cap.get(cv2.CAP_PROP_FPS)
fcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')  # cv2.VideoWriter_fourcc(*'DIVX')
out_video = cv2.VideoWriter('/Users/sunghun/Desktop/10.mp4', fcc, fps, (width, height))

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# map = folium.Map(location=[37.619774, 127.060926])
# folium.Marker([37.619774, 127.060926]).add_to(map)

map = cv2.imread('/Users/sunghun/Desktop/map.png')
map = cv2.resize(map,(2560,720))

while (cap.isOpened()):

    ret, frame = cap.read()

    if not ret:
        break

    rgb_frame = frame[:, :, ::-1]
    # frame preprocessing
    real = frame.copy()

    ori_imgs, framed_imgs, framed_metas = preprocess_video(frame, max_size=input_size)

    if use_cuda:
        x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
    else:
        x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)

    x = x.to(torch.float32 if not use_float16 else torch.float16).permute(0, 3, 1, 2)

    # model predict
    with torch.no_grad():
        features, regression, classification, anchors = model(x)

        out = postprocess(x,
                        anchors, regression, classification,
                        regressBoxes, clipBoxes,
                        threshold, iou_threshold)

    if len(out[0]['rois']) == 0:
        img = np.hstack((real,frame))
        img = np.vstack((img,map))
        cv2.imshow('frame', img)
        out_video.write(frame)
        cv2.imshow('map',map)
        continue

    # result

    out = invert_affine(framed_metas, out)
    # img_show = display(out, ori_imgs,"no",imshow=True)
    img_show = display2(out, ori_imgs)

    # show frame by frame
    img = np.hstack((real, img_show))
    img = np.vstack((img,map))
    cv2.imshow('frame', img)
    # cv2.imshow('frame',img_show)
    out_video.write(img_show)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out_video.release()
cv2.destroyAllWindows()