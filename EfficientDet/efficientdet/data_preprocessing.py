import torchvision.transforms

from .transforms import *
from .custom_dataset import Resizer,Augmenter,Normalizer

class TrainAugmentation:
    def __init__(self, size, mean=0, std=1.0):
        """
        Args:
            size: the size the of final image.
            mean: mean pixel value per channel.
        """
        self.mean = mean
        self.size = size
        self.std = std
        self.augment = Compose([
            PhotometricDistort(),
            Expand(self.mean),
            RandomSampleCrop(),
            RandomRotate(),
            Normalizer(mean=self.mean, std=self.std),
            Augmenter(), # Mirroring
            Resizer(self.size),
        ])

    def __call__(self, sample):
        """

        Args:
            img: the output of cv.imread in RGB layout.
            boxes: boundding boxes in the form of (x1, y1, x2, y2).
            labels: labels of boxes.
        """
        return self.augment(sample)


"""
test 같은 경우는 이미지를 돌리고, 자르고 , 색도 바꾸고 하면서 test 하지만,
train이나, prediction 같은 경우에는 resize랑 평균빼기(전처리) 정도만 해주는 걸로 보인다.
"""