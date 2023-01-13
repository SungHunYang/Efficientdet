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

from utils.utils import preprocess, invert_affine, postprocess, preprocess_video
from efficientdet.utils import BBoxTransform, ClipBoxes

compound_coef = 0
force_input_size = None  # set None to use default size

# replace this part with your project's anchor config
anchor_ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
anchor_scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]

threshold = 0.2
iou_threshold = 0.2

use_cuda = False
use_float16 = False
cudnn.fastest = True
cudnn.benchmark = True

obj_list = ['BACKGOUND', 'ConcreteCrack', 'Exposure', 'Spalling', 'PaintDamage', 'Efflorescene', 'SteelDefect']

# tf bilinear interpolation is different from any other's, just make do
input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536]
input_size = input_sizes[compound_coef] if force_input_size is None else force_input_size


model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list),
                             ratios=anchor_ratios, scales=anchor_scales)
model.load_state_dict(torch.load(f'./efficientdet-d0_26_21924_0.5826629481312854_20221209.pth', map_location=torch.device('cpu')))
model.requires_grad_(False)
model.eval()

if use_cuda:
    model = model.cuda()
if use_float16:
    model = model.half()



def speedcheck(x):

    print('running speed test...')
    with torch.no_grad():
        print('test1: model inferring and postprocessing')
        print('inferring image for 10 times...')
        t1 = time.time()
        for _ in range(10):
            _, regression, classification, anchors = model(x)

            out = postprocess(x,
                              anchors, regression, classification,
                              regressBoxes, clipBoxes,
                              threshold, iou_threshold)
            out = invert_affine(framed_metas, out)

        t2 = time.time()
        tact_time = (t2 - t1) / 10
        print(f'{tact_time} seconds, {1 / tact_time} FPS, @batch_size 1')

        """
    # uncomment this if you want a extreme fps test
    print('test2: model inferring only')
    print('inferring images for batch_size 32 for 10 times...')
    t1 = time.time()
    x = torch.cat([x] * 32, 0)
    for _ in range(10):
        _, regression, classification, anchors = model(x)

    t2 = time.time()
    tact_time = (t2 - t1) / 10
    print(f'{tact_time} seconds, {32 / tact_time} FPS, @batch_size 32')
        """

def display(preds, imgs, name, imshow=False, imwrite=False):
    colors = [
        (0, 0, 255),  # red
        (150, 62, 255),  # violetred
        (255, 0, 255),  # magenta
        (250, 206, 135),  # lightskyblue
        (127, 255, 0),  # springgreen
        (0, 165, 255),  # orange
    ]
    text_color = (255, 255, 255)

    for i in range(len(imgs)):
        if len(preds[i]['rois']) == 0:
            # continue
            return imgs

        for j in range(len(preds[i]['rois'])):
            (x1, y1, x2, y2) = preds[i]['rois'][j].astype(np.int32)

            obj = obj_list[preds[i]['class_ids'][j]]
            score = float(preds[i]['scores'][j])
            text = f"{obj}: {score:.2f}"
            (w, h), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            for k in range(len(obj_list)):
                if obj == obj_list[0]:
                    cv2.rectangle(imgs[i], (x1, y1), (x2, y2), (255, 255, 255), 2)
                    cv2.rectangle(imgs[i], (x1, y1 - 20), (x1 + w, y1), (255, 255, 255), -1)
                    cv2.putText(imgs[i], text,(x1 , y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (255, 255, 255), 2) # cv2.FONT_HERSHEY_SIMPLEX
                elif obj == obj_list[k] and k != 0:
                    cv2.rectangle(imgs[i], (x1, y1), (x2, y2), colors[k-1], 2)
                    cv2.rectangle(imgs[i], (x1, y1 - 20), (x1 + w, y1), colors[k-1], -1)
                    cv2.putText(imgs[i], text,(x1 , y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (255, 255, 255), 2) # cv2.FONT_HERSHEY_SIMPLEX
                else:
                    continue

        if imwrite:
            imgs[i] = cv2.resize(imgs[i],(512,512))
            cv2.imwrite(f'{out_path}/{name}', imgs[i])

    if imshow:
        return imgs


def display2(preds, imgs):
    for i in range(len(imgs)):
        if len(preds[i]['rois']) == 0:
            continue

        for j in range(len(preds[i]['rois'])):
            (x1, y1, x2, y2) = preds[i]['rois'][j].astype(np.int)
            cv2.rectangle(imgs[i], (x1, y1), (x2, y2), (255, 255, 0), 2)
            obj = obj_list[preds[i]['class_ids'][j]]
            score = float(preds[i]['scores'][j])

            cv2.putText(imgs[i], '{}, {:.3f}'.format(obj, score),
                        (x1, y1 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 0), 1)

        return imgs[i]

regressBoxes = BBoxTransform()
clipBoxes = ClipBoxes()

# cap = cv2.VideoCapture('/Users/sunghun/Desktop/capstone/video/1.mp4')
# width = int(cap.get(3)) # 가로 길이 가져오기
# height = int(cap.get(4)) # 세로 길이 가져오기
# fps = 20
# fcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
# out = cv2.VideoWriter('/Users/sunghun/Desktop/capstone/video/1_1.mp4', fcc, fps, (width, height))

# Video capture
cap = cv2.VideoCapture(1)

while True:

    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = frame[:, :, ::-1]

    # frame preprocessing

    start = time.time()
    ori_imgs, framed_imgs, framed_metas = preprocess_video(frame, max_size=input_size)
    print(f"preprocessing time : {time.time()-start}")

    start = time.time()
    if use_cuda:
        x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
    else:
        x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)

    x = x.to(torch.float32 if not use_float16 else torch.float16).permute(0, 3, 1, 2)
    print(f"stack time : {time.time()-start}")


    # model predict
    with torch.no_grad():
        start = time.time()
        features, regression, classification, anchors = model(x)
        print(f"model time : {time.time()-start}")

        start = time.time()
        out = postprocess(x,
                        anchors, regression, classification,
                        regressBoxes, clipBoxes,
                        threshold, iou_threshold)
        print(f"postprocessing : {time.time()-start}")

    if len(out[0]['rois']) == 0:
        cv2.imshow('frame',frame)
        continue

    # result
    start = time.time()
    out = invert_affine(framed_metas, out)
    print(f"affine time : {time.time()-start}")
    # img_show = display(out, ori_imgs,"no",imshow=True)

    start = time.time()
    img_show = display2(out, ori_imgs)
    print(f"display time : {time.time()-start}")

    # show frame by frame
    cv2.imshow('frame',img_show)

    if cv2.waitKey(10000) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()