# from https://github.com/amdegroot/ssd.pytorch


import torch
from torchvision import transforms
import cv2
import numpy as np
import types
from numpy import random


def intersect(box_a, box_b):
    max_xy = np.minimum(box_a[:, 2:], box_b[2:])
    min_xy = np.maximum(box_a[:, :2], box_b[:2])
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf) # box_a 는 default 인거 같고, box_b는 truth
    """
    numpy.clip(array, min, max)
    array 내의 element들에 대해서
    min 값 보다 작은 값들을 min값으로 바꿔주고
    max 값 보다 큰 값들을 max값으로 바꿔주는 함수
    """
    return inter[:, 0] * inter[:, 1] # 두 box의 교집합 -> 겹치는 부분의 넓이


def jaccard_numpy(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: Multiple bounding boxes, Shape: [num_boxes,4]
        box_b: Single bounding box, Shape: [4]
    Return:
        jaccard overlap: Shape: [box_a.shape[0], box_a.shape[1]]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1]))  # [A,B]
    area_b = ((box_b[2]-box_b[0]) *
              (box_b[3]-box_b[1]))  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]  # IOU 인거 같던데


class Compose(object):
    """Composes several augmentations together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
        >>> augmentations.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample


class Lambda(object):
    """Applies a lambda as a transform."""

    def __init__(self, lambd):
        assert isinstance(lambd, types.LambdaType) # assert 오류 찾아주기
        self.lambd = lambd

    def __call__(self, img, boxes=None, labels=None):
        return self.lambd(img, boxes, labels)

class SubtractMeans(object): # 평균 빼기? # 전처리 과정
    def __init__(self, mean):
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image, boxes=None, labels=None):
        image = image.astype(np.float32)
        image -= self.mean
        return image.astype(np.float32), boxes, labels #RGB 컬러 각각 평균빼기 안하고, 전체 평균에서 뺐다.


class ToAbsoluteCoords(object):  # 좌표를 실제 값으로 맵핑  -> 절대좌표
    def __call__(self, image, boxes=None, labels=None):
        height, width, channels = image.shape
        boxes[:, 0] *= width
        boxes[:, 2] *= width
        boxes[:, 1] *= height
        boxes[:, 3] *= height

        return image, boxes, labels
"""
여길 보면 boxes는 구성 요소가 xmin, ymin, xmax, ymax 임을 알 수 있다.

boxes 랑 locations 는 다른것 같음
"""

class ToPercentCoords(object): # 좌표를 비율로 맵핑
    def __call__(self, image, boxes=None, labels=None):
        height, width, channels = image.shape
        boxes[:, 0] /= width
        boxes[:, 2] /= width
        boxes[:, 1] /= height
        boxes[:, 3] /= height

        return image, boxes, labels


class Resize(object): # 300 X 300 size
    def __init__(self, size=300):
        self.size = size

    def __call__(self, image, boxes=None, labels=None):
        image = cv2.resize(image, (self.size,
                                 self.size))
        return image, boxes, labels


class RandomSaturation(object): # 이미지 채도 랜덤 조정 -> 색이 탁하고 흐림
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, sample):
        if random.randint(2): # 0,1 50% 확률
            image, annots = sample['img'], sample['annot']
            image[:, :, 1] *= random.uniform(self.lower, self.upper)
            sample = {'img': image, 'annot': annots}

        return sample


class RandomHue(object): # 이미지 색상 랜덤 조정 -> 5가지 보색으로 뭔가 조절?
    def __init__(self, delta=18.0):
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta

    def __call__(self, sample): # Hue(색상) 은 0°에서 360° 사이로 표현
        if random.randint(2):
            image, annots = sample['img'], sample['annot']
            image[:, :, 0] += random.uniform(-self.delta, self.delta)  # 랜덤한 값을 더해 보색 만들기
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0 # 저건 360° 안에 들어오는 값으로 바꾸겠다는 뜻
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0
            sample = {'img': image, 'annot': annots}
        return sample


class RandomLightingNoise(object): # 이미지의 H W C 의 위치를 random 조정
    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))

    def __call__(self, sample):
        if random.randint(2):
            swap = self.perms[random.randint(len(self.perms))]
            shuffle = SwapChannels(swap)  # shuffle channels
            image, annots = sample['img'], sample['annot']
            image = shuffle(image)

            sample = {'img': image, 'annot': annots}

        return sample


class ConvertColor(object):
    def __init__(self, current, transform):
        self.transform = transform
        self.current = current

    # HSV -> 0°에서 360° , RGB -> 0 ~ 256의 값
    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']
        if self.current == 'BGR' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)    # cvtColor -> convert color
        elif self.current == 'RGB' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        elif self.current == 'BGR' and self.transform == 'RGB':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif self.current == 'HSV' and self.transform == 'BGR':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        elif self.current == 'HSV' and self.transform == "RGB":
            image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
        else:
            raise NotImplementedError
        sample = {'img': image, 'annot': annots}
        return sample


class RandomContrast(object): # 색의 대비 랜덤 조절  -> 내생각 이렇게 하면서 데이터를 늘리는 듯
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    # expects float image
    def __call__(self, sample):
        if random.randint(2):
            image, annots = sample['img'], sample['annot']
            alpha = random.uniform(self.lower, self.upper)
            image *= alpha # 일정한 값을 곱하면, 1 이상 -> 큰값은 더 크게, 작은 값은 더 작게, 1이하 -> 반대 대비가 작아지겠지
            sample = {'img': image, 'annot': annots}
        return sample


class RandomBrightness(object): # 전체적으로 랜덤으로 밝아지게 하는
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, sample):
        if random.randint(2):
            image, annots = sample['img'], sample['annot']
            delta = random.uniform(-self.delta, self.delta)
            image += delta
            sample = {'img': image, 'annot': annots}
        return sample


class ToCV2Image(object):
    def __call__(self, tensor, boxes=None, labels=None):
        return tensor.cpu().numpy().astype(np.float32).transpose((1, 2, 0)), boxes, labels
# 얘는 C H W  -> H W C 로 바꾸는것

class ToTensor(object):
    def __call__(self, cvimage, boxes=None, labels=None):
        return torch.from_numpy(cvimage.astype(np.float32)).permute(2, 0, 1), boxes, labels
# 얘는 H W C -> C H W 로 바꾸는것

# Tensor 폼이랑 CV2 image 폼으로 바꾸는 것

class RandomSampleCrop(object): # 이미지 Crop
    """Crop
    Arguments:
        img (Image): the image being input during training
        boxes (Tensor): the original bounding boxes in pt form
        labels (Tensor): the class labels for each bbox
        mode (float tuple): the min and max jaccard overlaps
    Return:
        (img, boxes, classes)
            img (Image): the cropped image
            boxes (Tensor): the adjusted bounding boxes in pt form
            labels (Tensor): the class labels for each bbox
    """
    def __init__(self):
        self.sample_options = (
            # using entire original input image
            None,
            # sample a patch s.t. MIN jaccard w/ obj in .1,.3,.4,.7,.9
            (0.1, None),
            (0.3, None),
            (0.7, None),
            (0.9, None),
            # randomly sample a patch
            (None, None),
        )

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']
        height, width, _ = image.shape
        while True:
            # randomly choose a mode
            mode = random.choice(self.sample_options)
            if mode is None:
                return sample

            min_iou, max_iou = mode
            if min_iou is None:
                min_iou = float('-inf')
            if max_iou is None:
                max_iou = float('inf')

            # max trails (50)
            for _ in range(50):
                current_image = image

                w = random.uniform(0.3 * width, width)
                h = random.uniform(0.3 * height, height)

                # aspect ratio constraint b/t .5 & 2
                if h / w < 0.5 or h / w > 2:
                    continue

                left = random.uniform(width - w)
                top = random.uniform(height - h)

                # convert to integer rect x1,y1,x2,y2
                rect = np.array([int(left), int(top), int(left+w), int(top+h)])

                # calculate IoU (jaccard overlap) b/t the cropped and gt boxes
                overlap = jaccard_numpy(annots[:, :4], rect)

                # is min and max overlap constraint satisfied? if not try again
                if overlap.min() < min_iou and max_iou < overlap.max():
                    continue

                # cut the crop from the image
                current_image = current_image[rect[1]:rect[3], rect[0]:rect[2],
                                              :] # 구해진 직사각형으로 image crop

                # keep overlap with gt box IF center in sampled patch
                centers = (annots[:, :2] + annots[:, 2:4]) / 2.0 # 원래 box 중간값 찾기

                # mask in all gt boxes that above and to the left of centers
                m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1]) # and 대신 * 쓴거 같은 느낌?

                # mask in all gt boxes that under and to the right of centers
                m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])

                # mask in that both m1 and m2 are true
                mask = m1 * m2

                # have any valid boxes? try again if not
                if not mask.any():  # 완벽하게 들어 맞는다면
                    continue

                # take only matching gt boxes
                # current_boxes = annots[mask, :].copy() # 겹치는 부분에 대해 crop한 것을 copy


                # take only matching gt labels
                # annots[:, 4] = annots[mask, 4]

                # should we use the box left and top corner or the crop's
                annots[mask, :2] = np.maximum(annots[mask, :2],
                                                  rect[:2])
                # adjust to crop (by substracting crop's left,top)
                annots[mask, :2] -= rect[:2]

                annots[mask, 2:4] = np.minimum(annots[mask, 2:4],
                                                  rect[2:])
                # adjust to crop (by substracting crop's left,top)
                annots[mask, 2:4] -= rect[:2]

                sample = {'img': current_image, 'annot': annots[mask]}

                return sample


class Expand(object):
    def __init__(self, mean):
        self.mean = mean

    def __call__(self, sample):
        if random.randint(2):
            return sample
        image, annots = sample['img'], sample['annot']
        height, width, depth = image.shape
        ratio = random.uniform(1, 4)
        left = random.uniform(0, width*ratio - width)
        top = random.uniform(0, height*ratio - height)

        expand_image = np.zeros(
            (int(height*ratio), int(width*ratio), depth),
            dtype=image.dtype)
        expand_image[:, :, :] = self.mean
        expand_image[int(top):int(top + height),
                     int(left):int(left + width)] = image # ????
        image = expand_image

        boxes = annots.copy()
        boxes[:, :2] += (int(left), int(top))
        boxes[:, 2:4] += (int(left), int(top))

        # 확대된 만큼 box size도 키워줌
        sample = {'img': image, 'annot': boxes}
        return sample


class SwapChannels(object):
    """Transforms a tensorized image by swapping the channels in the order
     specified in the swap tuple.
    Args:
        swaps (int triple): final order of channels
            eg: (2, 1, 0)
    """

    def __init__(self, swaps):
        self.swaps = swaps

    def __call__(self, image):
        """
        Args:
            image (Tensor): image tensor to be transformed
        Return:
            a tensor with channels swapped according to swap
        """
        # if torch.is_tensor(image):
        #     image = image.data.cpu().numpy()
        # else:
        #     image = np.array(image)
        image = image[:, :, self.swaps]
        return image


class PhotometricDistort(object): # 그동안 했던거 여러개 모아서 해버리는
    def __init__(self):
        self.pd = [
            RandomContrast(),  # RGB
            ConvertColor(current="RGB", transform='HSV'),  # HSV
            RandomSaturation(),  # HSV
            RandomHue(),  # HSV
            ConvertColor(current='HSV', transform='RGB'),  # RGB
            RandomContrast()  # RGB
        ]
        self.rand_brightness = RandomBrightness()
        self.rand_light_noise = RandomLightingNoise()

    def __call__(self, sample):
        sample = self.rand_brightness(sample)
        if random.randint(2):
            distort = Compose(self.pd[:-1])
        else:
            distort = Compose(self.pd[1:])
        sample = distort(sample)
        return self.rand_light_noise(sample)

# RANDOM ROTATION CLASS and FUNCTIONS BELONGING TO THE CLASS
def rotate_im(image, angle):
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    image = cv2.warpAffine(image, M, (nW, nH))

    #    image = cv2.resize(image, (w,h))
    return image

def get_corners(bboxes):
    width = (bboxes[:, 2] - bboxes[:, 0]).reshape(-1, 1)
    height = (bboxes[:, 3] - bboxes[:, 1]).reshape(-1, 1)
    x1 = bboxes[:, 0].reshape(-1, 1)
    y1 = bboxes[:, 1].reshape(-1, 1)
    x2 = x1 + width
    y2 = y1
    x3 = x1
    y3 = y1 + height
    x4 = bboxes[:, 2].reshape(-1, 1)
    y4 = bboxes[:, 3].reshape(-1, 1)
    corners = np.hstack((x1, y1, x2, y2, x3, y3, x4, y4))
    return corners


def rotate_box(corners, angle, cx, cy, h, w):
    corners = corners.reshape(-1, 2)
    corners = np.hstack((corners, np.ones((corners.shape[0], 1), dtype=type(corners[0][0]))))
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    M[0, 2] += (nW / 2) - cx
    M[1, 2] += (nH / 2) - cy

    calculated = np.dot(M, corners.T).T
    calculated = calculated.reshape(-1, 8)
    return calculated

def get_enclosing_box(corners):
    x_ = corners[:, [0, 2, 4, 6]]
    y_ = corners[:, [1, 3, 5, 7]]
    xmin = np.min(x_, 1).reshape(-1, 1)
    ymin = np.min(y_, 1).reshape(-1, 1)
    xmax = np.max(x_, 1).reshape(-1, 1)
    ymax = np.max(y_, 1).reshape(-1, 1)
    final = np.hstack((xmin, ymin, xmax, ymax, corners[:, 8:]))
    return final

def bbox_area(bbox):
    return (bbox[:, 2] - bbox[:, 0]) * (bbox[:, 3] - bbox[:, 1])

def clip_box(bbox, clip_box, alpha):
    ar_ = (bbox_area(bbox))
    x_min = np.maximum(bbox[:, 0], clip_box[0]).reshape(-1, 1)
    y_min = np.maximum(bbox[:, 1], clip_box[1]).reshape(-1, 1)
    x_max = np.minimum(bbox[:, 2], clip_box[2]).reshape(-1, 1)
    y_max = np.minimum(bbox[:, 3], clip_box[3]).reshape(-1, 1)

    bbox = np.hstack((x_min, y_min, x_max, y_max, bbox[:, 4:]))
    delta_area = ((ar_ - bbox_area(bbox)) / ar_)
    mask = (delta_area < (1 - alpha)).astype(int)
    bbox = bbox[mask == 1, :]
    return bbox

class RandomRotate(object):
    def __init__(self, angle=10): # standard rotation angle = 20degrees
        self.angle = angle
        if type(self.angle) == tuple:
            assert len(self.angle) == 2 # if given range more than 2
        else:
            self.angle = (-self.angle, self.angle)
    def __call__(self, sample):

        if random.randint(2):
            return sample # 무조건 lotate 하게 만듦
        image, annots = sample['img'], sample['annot']
        boxes = annots[:,:4]
        angle = random.uniform(*self.angle)
        w, h = image.shape[1], image.shape[0]
        cx, cy = w // 2, h // 2
        image = rotate_im(image, angle)
        corners = get_corners(boxes)
        corners = np.hstack((corners, boxes[:, 4:]))
        corners[:, :8] = rotate_box(corners[:, :8], angle, cx, cy, h, w)
        new_boxes = get_enclosing_box(corners)
        scale_factor_x = image.shape[1] / w
        scale_factor_y = image.shape[0] / h
        image = cv2.resize(image, (w, h))
        new_boxes[:, :4] /= [scale_factor_x, scale_factor_y, scale_factor_x, scale_factor_y]
        boxes = new_boxes
        boxes = clip_box(boxes, [0, 0, w, h], 0.25)
        annots[:,:4] = boxes

        sample = {'img': image, 'annot': annots}

        return sample
