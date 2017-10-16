import pickle
import numpy as np
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


def load_all_train_imgs_ids():
    with open(config.IMG_IDS_PATH, 'rb') as f:
        res = pickle.load(f)
    return res


class CdiscountDataset(Dataset):
    def __init__(self, img_ids, mode, transform=None):
        """
        :param img_ids: list of tuples (product_id, img_number),
        img_number between 0 and 3
        """
        assert mode in ('train', 'test')
        # assert isinstance(img_ids, np.array)
        self._img_ids = img_ids
        self._dataset_length = None
        self._transform = transform
        self._client = MongoClient(connect=False)
        self._db = self._client.cdiscount[mode]
        self._cat_to_label = {v: k for k, v in LABEL_TO_CAT.items()}

    def __len__(self):
        return self._calculate_length()

    def __getitem__(self, item):
        product_id, image_number = self._img_ids[item]
        product = self._db.find_one({'_id': product_id})
        img = self._img_from_bytes(product['imgs'][image_number]['picture'])
        if self._transform is not None:
            img = self._transform(img)
        label = self._cat_to_label[product['category_id']]
        return img, label

    def _calculate_length(self):
        if self._dataset_length is not None:
            return self._dataset_length
        self._dataset_length = len(self._img_ids)
        return self._dataset_length

    @staticmethod
    def _img_from_bytes(img_bytes):
        np_img_str = np.fromstring(img_bytes, np.uint8)
        img = cv2.imdecode(np_img_str, cv2.IMREAD_COLOR)
        assert img is not None
        # img = img.astype('float') / 255.
        return img
