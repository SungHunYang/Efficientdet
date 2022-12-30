import os
import glob
import cv2
import json
from pathlib import Path
from tqdm import tqdm

Image_path = Path('/Users/sunghun/Desktop/capstone/data')
json_path = Path('/Users/sunghun/Desktop/capstone/data/json')

names = ['new_con','new_des','new_fe']

### 간단하게 중복 확인하는 방식으로 set 사용

json_set = set( _.split('.')[0] for _ in os.listdir(json_path) )
image_set = set(_.split('.')[0] for _ in os.listdir(f"{Image_path}/{names[2]}"))


print(len(json_set))
print(len(image_set))
print(len(image_set-json_set))