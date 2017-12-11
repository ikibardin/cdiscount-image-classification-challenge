import numpy as np
import math
import torch
from torchvision import transforms
from torchvision.transforms import Lambda, ToTensor
import cv2
import numbers
import random

try:
    import accimage
except ImportError:
    accimage = None
from PIL import Image


def _is_pil_image(img):
    if accimage is not None:
        return isinstance(img, (Image.Image, accimage.Image))
    else:
        return isinstance(img, Image.Image)


def _hflip(img):
    """Horizontally flip the given PIL Image.
    Args:
        img (PIL Image): Image to be flipped.
    Returns:
        PIL Image:  Horizontall flipped image.
    """
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))
    return img.transpose(Image.FLIP_LEFT_RIGHT)


def _vflip(img):
    """Vertically flip the given PIL Image.
    Args:
        img (PIL Image): Image to be flipped.
    Returns:
        PIL Image:  Vertically flipped image.
    """
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))
    return img.transpose(Image.FLIP_TOP_BOTTOM)


def _crop(img, i, j, h, w):
    """Crop the given PIL Image.
    Args:
        img (PIL Image): Image to be cropped.
        i: Upper pixel coordinate.
        j: Left pixel coordinate.
        h: Height of the cropped image.
        w: Width of the cropped image.
    Returns:
        PIL Image: Cropped image.
    """
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    return img.crop((j, i, j + w, i + h))


def _center_crop(img, output_size):
    if isinstance(output_size, numbers.Number):
        output_size = (int(output_size), int(output_size))
    w, h = img.size
    th, tw = output_size
    i = int(round((h - th) / 2.))
    j = int(round((w - tw) / 2.))
    return _crop(img, i, j, th, tw)


def _five_crop(img, size):
    """Crop the given PIL Image into four corners and the central crop.
    .. Note::
        This transform returns a tuple of images and there may be a
        mismatch in the number of inputs and targets your ``Dataset`` returns.
    Args:
       size (sequence or int): Desired output size of the crop. If size is an
           int instead of sequence like (h, w), a square crop (size, size) is
           made.
    Returns:
        tuple: tuple (tl, tr, bl, br, center) corresponding top left,
            top right, bottom left, bottom right and center crop.
    """
    if isinstance(size, numbers.Number):
        size = (int(size), int(size))
    else:
        assert len(
            size) == 2, "Please provide only two dimensions (h, w) for size."

    w, h = img.size
    crop_h, crop_w = size
    if crop_w > w or crop_h > h:
        raise ValueError(
            "Requested crop size {} is bigger than input size {}".format(size,
                                                                         (h,
                                                                          w)))
    tl = img.crop((0, 0, crop_w, crop_h))
    tr = img.crop((w - crop_w, 0, w, crop_h))
    bl = img.crop((0, h - crop_h, crop_w, h))
    br = img.crop((w - crop_w, h - crop_h, w, h))
    center = _center_crop(img, (crop_h, crop_w))
    return tl, tr, bl, br, center


def _ten_crop(img, size, vertical_flip=False):
    """Crop the given PIL Image into four corners and the central crop plus the
       flipped version of these (horizontal flipping is used by default).
    .. Note::
        This transform returns a tuple of images and there may be a
        mismatch in the number of inputs and targets your ``Dataset`` returns.
       Args:
           size (sequence or int): Desired output size of the crop. If size is an
               int instead of sequence like (h, w), a square crop (size, size) is
               made.
           vertical_flip (bool): Use vertical flipping instead of horizontal
        Returns:
            tuple: tuple (tl, tr, bl, br, center, tl_flip, tr_flip, bl_flip,
                br_flip, center_flip) corresponding top left, top right,
                bottom left, bottom right and center crop and same for the
                flipped image.
    """
    if isinstance(size, numbers.Number):
        size = (int(size), int(size))
    else:
        assert len(
            size) == 2, "Please provide only two dimensions (h, w) for size."

    first_five = _five_crop(img, size)

    if vertical_flip:
        img = _vflip(img)
    else:
        img = _hflip(img)

    second_five = _five_crop(img, size)
    return first_five + second_five


class TenCrop(object):
    """Crop the given PIL Image into four corners and the central crop plus the flipped version of
    these (horizontal flipping is used by default)
    .. Note::
         This transform returns a tuple of images and there may be a mismatch in the number of
         inputs and targets your Dataset returns. See below for an example of how to deal with
         this.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        vertical_flip(bool): Use vertical flipping instead of horizontal
    Example:
         >>> transform = Compose([
         >>>    TenCrop(size), # this is a list of PIL Images
         >>>    Lambda(lambda crops: torch.stack([ToTensor()(crop) for crop in crops])) # returns a 4D tensor
         >>> ])
         >>> #In your test loop you can do the following:
         >>> input, target = batch # input is a 5d tensor, target is 2d
         >>> bs, ncrops, c, h, w = input.size()
         >>> result = model(input.view(-1, c, h, w)) # fuse batch size and ncrops
         >>> result_avg = result.view(bs, ncrops, -1).mean(1) # avg over crops
    """

    def __init__(self, size, vertical_flip=False):
        self.size = size
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            assert len(
                size) == 2, "Please provide only two dimensions (h, w) for size."
            self.size = size
        self.vertical_flip = vertical_flip

    def __call__(self, img):
        return _ten_crop(img, self.size, self.vertical_flip)


def tta_transform(norm_mean, norm_std, crop=TenCrop(160)):
    return transforms.Compose([
        transforms.ToPILImage(),
        crop,
        Lambda(lambda crops: torch.stack(
            [transforms.Normalize(mean=norm_mean,
                                  std=norm_std)(ToTensor()(crop))
             for crop in crops]))
    ])


def no_tta_trans(norm_mean, norm_std):
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean,
                             std=norm_std)
    ])


def random_shift_scale_rotate(image, shift_limit=[-0.0625, 0.0625],
                              scale_limit=[1 / 1.2, 1.2],
                              rotate_limit=[-15, 15], aspect_limit=[1, 1],
                              size=[-1, -1], borderMode=cv2.BORDER_REFLECT_101,
                              u=0.5):
    # cv2.BORDER_REFLECT_101  cv2.BORDER_CONSTANT

    if random.random() < u:
        height, width, channel = image.shape
        if size[0] == -1: size[0] = width
        if size[1] == -1: size[1] = height

        angle = random.uniform(rotate_limit[0], rotate_limit[1])  # degree
        scale = random.uniform(scale_limit[0], scale_limit[1])
        aspect = random.uniform(aspect_limit[0], aspect_limit[1])
        sx = scale * aspect / (aspect ** 0.5)
        sy = scale / (aspect ** 0.5)
        dx = round(random.uniform(shift_limit[0], shift_limit[1]) * width)
        dy = round(random.uniform(shift_limit[0], shift_limit[1]) * height)

        cc = math.cos(angle / 180 * math.pi) * (sx)
        ss = math.sin(angle / 180 * math.pi) * (sy)
        rotate_matrix = np.array([[cc, -ss], [ss, cc]])

        box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
        box1 = box0 - np.array([width / 2, height / 2])
        box1 = np.dot(box1, rotate_matrix.T) + np.array(
            [width / 2 + dx, height / 2 + dy])

        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0, box1)

        image = cv2.warpPerspective(image, mat, (size[0], size[1]),
                                    flags=cv2.INTER_LINEAR,
                                    borderMode=borderMode, borderValue=(0, 0,
                                                                        0,))  # cv2.BORDER_CONSTANT, borderValue = (0, 0, 0))  #cv2.BORDER_REFLECT_101

    return image


def random_horizontal_flip(image, u=0.5):
    if random.random() < u:
        image = cv2.flip(image, 1)  # np.fliplr(img) ##left-right
    return image


def _pytorch_image_to_tensor_transform(image):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.transpose((2, 0, 1))
    tensor = torch.from_numpy(image).float().div(255)

    tensor[0] = (tensor[0] - mean[0]) / std[0]
    tensor[1] = (tensor[1] - mean[1]) / std[1]
    tensor[2] = (tensor[2] - mean[2]) / std[2]

    return tensor


def _image_to_tensor_transform(image):
    tensor = _pytorch_image_to_tensor_transform(image)
    tensor[0] = tensor[0] * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
    tensor[1] = tensor[1] * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
    tensor[2] = tensor[2] * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
    return tensor


def frogs_transform():
    return _image_to_tensor_transform


def train_augment(image):
    if random.random() < 0.5:
        image = random_shift_scale_rotate(image,
                                          # shift_limit  = [0, 0],
                                          shift_limit=[-0.06, 0.06],
                                          scale_limit=[0.9, 1.2],
                                          rotate_limit=[-10, 10],
                                          aspect_limit=[1, 1],
                                          # size=[1,299],
                                          borderMode=cv2.BORDER_REFLECT_101,
                                          u=1)
    else:
        pass

    # flip  random ---------
    image = random_horizontal_flip(image, u=0.5)

    tensor = _image_to_tensor_transform(image)
    return tensor


def _frogs_get_tta10(image):
    tensors = []
    for i in range(10):
        tensors.append(train_augment(image))
    return tensors


def frogs_tta():
    return transforms.Compose([
        _frogs_get_tta10,
        Lambda(lambda crops: torch.stack(crops))
    ])
