import json
import os
from pathlib import Path
import glob
import cv2
import numpy as np
import copy

json_path = Path('/Users/sunghun/Desktop/seg_data/json')
image_path = Path('/Users/sunghun/Desktop/seg_data/data')

out_path = Path('/Users/sunghun/Desktop/result') # image, black

images = glob.glob(f'{image_path}/*.jpg')

data_list = {}
for image in images:
    base_name = os.path.basename(image)
    json_name = base_name.split('.')[0]+'.json'

    ori = cv2.imread(image)
    H, W, _ = ori.shape
    black = np.zeros((H, W), dtype="uint8")

    data_list = {}
    with open(f"{json_path}/{json_name}",'rt',encoding='UTF8') as f:
        data = json.load(f)

        for i in range(len(data['Learning_Data_Info']['Annotations'])):
            a = data['Learning_Data_Info']['Annotations'][i]['polygon']
            result = []
            for j in range(0,len(a),2):
                result.append(a[j:j+2])

            data_list[i] = result

    for i in data_list:
        # for j in range(len(data_list[i])-1):
        #     cv2.line(black,data_list[i][j],data_list[i][j+1],(255),2)
        pt = np.array(data_list[i])
        cv2.fillPoly(black,[pt],(255)) # 색채우기 fillConvexPoly 라는 것도 있는데, 그건 꽉 안채워 지더라..

        hw = max(data_list[i])
        rc = min(data_list[i])

        if hw[0]+30 > H:
            hw[0] = H-30
        if hw[1]+30 > W:
            hw[1] = W-30
        if rc[0]-30 < 0:
            rc[0] = 31
        if rc[1]-30 < 0:
            rc[1] = 31

        H_min = rc[0]-30
        H_max = hw[0]+30
        W_min = rc[1]-30
        W_max = hw[1]+30

        q = copy.deepcopy(ori)
        e = copy.deepcopy(black)

        if H_min > H_max:
            continue
        if W_min > W_max:
            continue

        if H_max - H_min < 200:
            continue
        if W_max - W_min < 200:
            continue

        if (( H_max - H_min ) / ( W_max - W_min )) > 2 or (( W_max - W_min ) / ( H_max - H_min )) > 2:
            continue

        re = q[W_min:W_max, H_min:H_max]
        re_black = e[W_min:W_max, H_min:H_max]

        cv2.imwrite(f'{out_path}/image/{i}.{base_name}',re)
        cv2.imwrite(f'{out_path}/black/{i}.{base_name}',re_black)

cv2.imwrite(f'{out_path}/origin/{base_name}',ori)
cv2.imwrite(f'{out_path}/origin_black/{base_name}',black)