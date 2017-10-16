import numpy as np
import cv2
from pymongo import MongoClient
from torch.utils.data import Dataset
from tqdm import tqdm

from config import PRODUCT_COUNT


def load_all_imgs_ids(mode):
    assert mode in ('train', 'test')
    res = list()
    client = MongoClient(connect=False)
    db = client.cdiscount[mode]
    for product in tqdm(db.find(), total=PRODUCT_COUNT):
        assert 1 <= len(product['imgs']) <= 4
        for i in range(len(product['imgs'])):
            res.append((product['_id'], i))
    return res


class CdiscountDataset(Dataset):
    def __init__(self, img_ids, mode, transform=None):
        """
        :param img_ids: list of tuples (product_id, img_number),
        img_number between 0 and 3
        """
        assert mode in ('train', 'test')
        assert isinstance(img_ids, np.array)
        self._img_ids = img_ids
        self._dataset_length = None
        self._transform = transform
        self._client = MongoClient(connect=False)
        self._db = self._client.cdiscount[mode]

    def __len__(self):
        return self._calculate_length()

    def __getitem__(self, item):
        product_id, image_number = self._img_ids[item]
        product = self._db.find_one({'_id': product_id})
        img = self._img_from_bytes(product['imgs'][image_number]['picture'])
        if self._transform is not None:
            img = self._transform(img)
        label = product['category_id']
        return img, label

    def _calculate_length(self):
        if self._dataset_length is not None:
            return self._dataset_length
        self._dataset_length = len(self._img_ids)
        return self._dataset_length

    @staticmethod
    def _img_from_bytes(img_bytes):
        np_img_str = np.fromstring(img_bytes, np.uint8).astype('float') / 255.
        return cv2.imdecode(np_img_str, cv2.IMREAD_COLOR)
