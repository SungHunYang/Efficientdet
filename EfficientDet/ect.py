import cv2
import numpy as np
from efficientdet.data_preprocessing import TrainAugmentation
from efficientdet.custom_dataset import CustomDataset

img = cv2.imread('C:/Users/HP/Desktop/dogface_original_00428.jpg')
img = img.astype(np.float32) / 255.
annots =  np.array([[557,381,2645,2441,1]], dtype=np.float32)
sample = {'img':img,'annot':annots}

check = TrainAugmentation(512,0.0,1.)

r = check(sample)
image = r['img'].numpy()
annots = r['annot'].numpy()
scale = r['scale']
cv2.rectangle(image, (int(annots[0][0]), int(annots[0][1])), (int(annots[0][2]), int(annots[0][3])), (255, 255, 0), 2)
cv2.imshow('image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()


