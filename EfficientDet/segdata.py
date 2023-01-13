import json
import os
from pathlib import Path
import glob
import cv2
import numpy as np

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

    cv2.imwrite(f'{out_path}/black/{base_name}',black)

