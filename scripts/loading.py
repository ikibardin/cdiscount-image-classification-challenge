import pickle
import numpy as np
import pandas as pd
import cv2
from pymongo import MongoClient
from torch.utils.data import Dataset
from tqdm import tqdm

import config
from label_to_cat import LABEL_TO_CAT


def resize(img):
    res = cv2.resize(img, (299, 299))
    assert res is not None
    return res


def load_cat_to_big_label():
    df = pd.read_csv(config.CAT_TO_BIG_CAT_PATH)
    res = dict()
    for index, row in df.iterrows():
        res[row['cat_id']] = row['big_cat_id']
    return res


def load_all_train_imgs_ids():
    with open(config.IMG_IDS_PATH, 'rb') as f:
        res = pickle.load(f)
    return res


def load_separator_imgs_ids():
    with open(config.SEPARATOR_IMG_IDS_PATH_PICKLE, 'rb') as f:
        res = pickle.load(f)
    return res


def _img_from_bytes(img_bytes):
    np_img_str = np.fromstring(img_bytes, np.uint8)
    img = cv2.imdecode(np_img_str, cv2.IMREAD_COLOR)
    assert img is not None
    img = img.astype('float') / 255.
    return img


class CdiscountDataset(Dataset):
    def __init__(self, img_ids, mode, transform=None, big_cats=False):
        """
        :param img_ids: TRAIN: list of tuples (product_id, img_number),
        img_number between 0 and 3
        TEST: list of product_ids
        """
        assert mode in ('train', 'test')
        # assert isinstance(img_ids, np.array)
        self._img_ids = img_ids
        self._transform = transform
        self._client = MongoClient(connect=False)
        self._mode = mode
        self._db = self._client.cdiscount[mode]
        self._big_cats = big_cats
        if not self._big_cats:
            self._cat_to_label = {v: k for k, v in LABEL_TO_CAT.items()}
        else:
            self._cat_to_label = load_cat_to_big_label()

    def __len__(self):
        return len(self._img_ids)

    def __getitem__(self, item):
        """
        :param item:
        :return: In train mode -- tuple (img, label)
                 In test mode -- dict with '_id' and 'imgs' list
        """
        if self._mode == 'train':
            return self._load_item_train(item)
        elif self._mode == 'test':
            return self._load_item_test(item)
        else:
            raise ValueError('Unexpected mode.')

    def _load_item_train(self, item):
        assert self._mode == 'train'
        product_id, image_number = self._img_ids[item]
        product = self._db.find_one({'_id': product_id})
        img = _img_from_bytes(product['imgs'][image_number]['picture'])
        if self._transform is not None:
            img = self._transform(img)
        label = self._cat_to_label[product['category_id']]
        return img, label

    def _load_item_test(self, item):
        assert self._mode == 'test'
        raise RuntimeError('Not implemented')

# class CdiscountBigCategoriesDataset(Dataset):
