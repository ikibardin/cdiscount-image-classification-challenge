import numpy as np
import cv2
from pymongo import MongoClient
from torch.utils.data import Dataset


class CdiscountDataset(Dataset):
    def __init__(self, img_ids, mode, transform=None):
        """
        :param img_ids: list of tuples (product_id, img_number),
        img_number between 0 and 3
        """
        assert mode in ('train', 'test')
        self._img_ids = img_ids
        self._transform = transform
        self._client = MongoClient(connect=False)
        self._db = self._client.cdiscount[mode]

    def __len__(self):
        return len(self._img_ids)

    def __getitem__(self, item):
        product_id, image_number = self._img_ids[item]
        product = self._db.find_one({'_id': product_id})
        img = self._img_from_bytes(product['imgs'][image_number]['picture'])
        if self._transform is not None:
            img = self._transform(img)
        label = product['category_id']
        return img, label

    @staticmethod
    def _img_from_bytes(img_bytes):
        np_img_str = np.fromstring(img_bytes, np.uint8)
        return cv2.imdecode(np_img_str, cv2.IMREAD_COLOR)
