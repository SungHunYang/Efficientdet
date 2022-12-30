import os
import shutil
import json
from pathlib import Path
from tqdm import tqdm

train = Path('/Users/sunghun/Desktop/capstone/data/train.txt')
val = Path('/Users/sunghun/Desktop/capstone/data/val.txt')
json_path = Path('/Users/sunghun/Desktop/capstone/data/json_all')

conc_path = Path('/Users/sunghun/Desktop/capstone/data/new_con')
conc_val = {1: 1581, 2: 885, 3: 1260, 4: 992, 5: 889, 6: 117}

fe_path = Path('/Users/sunghun/Desktop/capstone/data/new_fe')
fe_val = {1: 676, 2: 509, 3: 420, 4: 310, 5: 247, 6: 75}

des_path = Path('/Users/sunghun/Desktop/capstone/data/new_des')
des_val = {1: 844, 2: 584, 3: 694, 4: 562, 5: 597, 6: 103}

cnt = {1:0,2:0,3:0,4:0,5:0,6:0}
for name in os.listdir(conc_path):
    json_name = name.split('.')[0]+'.json'

    with open(f"{json_path}/{json_name}", 'rt', encoding='UTF8') as f:
        data = json.load(f)

        cnt[len(data['annotations'])] += 1

        if cnt[len(data['annotations'])] <= conc_val[len(data['annotations'])]:
            with open(val,'a') as t:
                t.write(f"{name}\n")
        else:
            with open(train,'a') as t:
                t.write(f"{name}\n")

        shutil.move(f"{json_path}/{json_name}",f"/Users/sunghun/Desktop/capstone/data/json/{json_name}")

cnt = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
for name in os.listdir(des_path):
    json_name = name.split('.')[0] + '.json'

    with open(f"{json_path}/{json_name}", 'rt', encoding='UTF8') as f:
        data = json.load(f)

        cnt[len(data['annotations'])] += 1

        if cnt[len(data['annotations'])] <= des_val[len(data['annotations'])]:
            with open(val, 'a') as t:
                t.write(f"{name}\n")
        else:
            with open(train, 'a') as t:
                t.write(f"{name}\n")

        shutil.move(f"{json_path}/{json_name}", f"/Users/sunghun/Desktop/capstone/data/json/{json_name}")

cnt = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
for name in os.listdir(fe_path):
    json_name = name.split('.')[0] + '.json'

    with open(f"{json_path}/{json_name}", 'rt', encoding='UTF8') as f:
        data = json.load(f)

        cnt[len(data['annotations'])] += 1

        if cnt[len(data['annotations'])] <= fe_val[len(data['annotations'])]:
            with open(val, 'a') as t:
                t.write(f"{name}\n")
        else:
            with open(train, 'a') as t:
                t.write(f"{name}\n")

        shutil.move(f"{json_path}/{json_name}", f"/Users/sunghun/Desktop/capstone/data/json/{json_name}")

#
# for name in tqdm(os.listdir(fe_path)):
#     shutil.move(f"{fe_path}/{name}",f"/Users/sunghun/Desktop/capstone/data/total_512/{name}")


