import os
import io
import pickle
from PIL import Image
import numpy as np
import pandas as pd
import cv2
from pymongo import MongoClient
from torch.utils.data import Dataset

import config
from label_to_cat import LABEL_TO_CAT


def load_cat_to_big_label():
    df = pd.read_csv(config.CAT_TO_BIG_CAT_PATH, index_col=0)
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
    return img


class CdiscountDatasetPandas(Dataset):
    def __init__(self, img_ids_df, mode, transform=None, big_cats=False):
        """
        :param img_ids: TRAIN: list of tuples (product_id, img_number),
        img_number between 0 and 3
        TEST: list of product_ids
        """
        assert mode in ('train', 'test', 'valid')
        # assert isinstance(img_ids, np.array)
        self._img_ids = img_ids_df
        self._transform = transform
        self._client = MongoClient(connect=False)
        if mode == 'valid':
            self._db = self._client.cdiscount['train']
            self._mode = 'test'
        else:
            self._mode = mode
            self._db = self._client.cdiscount[mode]
        self._big_cats = big_cats
        if not self._big_cats:
            self._cat_to_label = {v: k for k, v in LABEL_TO_CAT.items()}
        else:
            self._cat_to_label = load_cat_to_big_label()

    def __len__(self):
        return self._img_ids.shape[0]

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
        _, _, img, product = self._load_all(item)
        label = self._cat_to_label[product['category_id']]
        return img, label

    def _load_item_test(self, item):
        assert self._mode == 'test'
        product_id, image_number, img, _ = self._load_all(item)
        return product_id, image_number, img

    def _load_all(self, index):
        product_id = self._img_ids.id.iloc[index]
        image_number = self._img_ids.image_numb.iloc[index]
        product_id = int(product_id)
        image_number = int(image_number)
        product = self._db.find_one({'_id': product_id})
        assert product is not None
        # print('%%%%%%%%%%%%%%%% Found', product_id)
        img = _img_from_bytes(product['imgs'][image_number]['picture'])
        if self._transform is not None:
            img = self._transform(img)
        return product_id, image_number, img, product


class ArturDataset(Dataset):
    def __init__(self, table, mode, transform=None):
        """
        :param img_ids: TRAIN: list of tuples (product_id, img_number),
        img_number between 0 and 3
        TEST: list of product_ids
        """
        assert mode in ('train', 'test')
        if mode == 'test':
            raise RuntimeError('Not implemented')
        # assert isinstance(img_ids, np.array)
        self._table = table
        self._transform = transform
        self._mode = mode
        self._cat_to_label = {v: k for k, v in LABEL_TO_CAT.items()}
        self._train_dir = '/media/bcache/generated_datasets/fattahov_cowc1_1/data/train'

    def __len__(self):
        return self._table.shape[0]

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
        image_name = self._table.image_name.iloc[item]
        cat = self._table.cat.iloc[item]
        # image_number = self._img_ids.image_numb[item]
        img = cv2.imread('{}/{}/{}'.format(self._train_dir,
                                           cat, image_name), cv2.IMREAD_COLOR)
        if self._transform is not None:
            img = self._transform(img)
        label = self._cat_to_label[cat]
        return img, label

    def _load_item_test(self, item):
        assert self._mode == 'test'
        raise RuntimeError('Not implemented')


VAL_BATCH = 2048


class StackingDataset(Dataset):
    def __init__(self, paths):
        self._paths = paths
        self._stores = []
        for path in self._paths:
            self._stores.append(pd.HDFStore(path))
        assert len(self._stores) > 0
        self._keys = self._stores[0].keys()
        for store in self._stores:
            assert len(self._keys) == len(store), \
                'keys len {}; store len {}'.format(len(self._keys),
                                                   len(store))
        self._len = (len(self._keys) - 1) * VAL_BATCH \
                    + self._stores[0].select(self._keys[-1]).shape[0]

    def features_count(self):
        return len(self._paths) * config.CAT_COUNT

    def __len__(self):
        return self._len

    def __getitem__(self, item):
        key_ind = item / VAL_BATCH
        tables = self._select_tables(self._keys[key_ind])
        raise NotImplementedError('xD')

    def _select_tables(self, key):
        tables = []
        for store in self._stores:
            tables.append(store.select(key))
        assert len(tables) == len(self._stores)
        return tables
