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


VAL_BATCH = 256


class StackingDataset(Dataset):
    def __init__(self, paths, meta_path, transform=None):
        self._storages = self._load_storages(paths)
        assert len(self._storages) > 0
        self._meta = pd.read_csv(meta_path)
        self._meta = self._meta[self._meta['train'] == 0.]
        self._length = self._meta.shape[0]
        self._transform = transform
        self._cat_to_label = {v: k for k, v in LABEL_TO_CAT.items()}

    def __len__(self):
        return self._length

    def __getitem__(self, item):
        key = 'd' + str(item / VAL_BATCH)
        tables = self._select_tables(key)
        index = item % VAL_BATCH
        id_, img_num = self._select_id_and_img_num(tables, index)

        features = []
        for table in tables:
            features.append(
                np.array(table.iloc[index].drop(['pr_id', 'img_num'],
                                                inplace=False)))
        features = np.hstack(features)
        if self._transform is not None:
            features = self._transform(features)
        tmp = self._meta[self._meta.id == id_]
        cat = tmp[tmp.image_numb == img_num].cat.iloc[0]  # topkek xD ))))
        label = self._cat_to_label[cat]
        return id_, img_num, features, label

    @staticmethod
    def _load_storages(paths):
        storages = []
        for path in paths:
            print("Loading storage at {}".format(path))
            storage = pd.HDFStore(path)
            print("Success, {} tables in storage.".format(len(storage)))
            storages.append(storage)
        return storages

    def _select_tables(self, key):
        tables = []
        for storage in self._storages:
            tables.append(storage.select(key))
        return tables

    @staticmethod
    def _select_id_and_img_num(tables, index):
        assert len(tables) > 0
        ids = []
        img_nums = []
        for table in tables:
            ids.append(table.pr_id.iloc[index])
            img_nums.append(table.img_num.iloc[index])
        id_ = ids[0]
        img_num = img_nums[0]
        for id1, num1 in zip(ids, img_nums):
            assert id_ == id1
            assert img_num == num1
        return id_, img_num


class StackingDatasetServer(Dataset):
    def __init__(self, paths, meta_path, transform=None):
        self._tables = self._load_tables(paths)
        self._meta = pd.read_csv(meta_path)
        self._meta = self._meta[self._meta['train'] == 0.]
        self._length = self._meta.shape[0]
        self._transform = transform
        self._cat_to_label = {v: k for k, v in LABEL_TO_CAT.items()}

    def __len__(self):
        return self._length

    def __getitem__(self, item):
        id_, img_num = self._select_id_and_img_num(item)
        features = []
        for table in self._tables:
            features.append(
                np.array(table.iloc[item].drop(['pr_id', 'img_num'],
                                               inplace=False)))
        features = np.hstack(features)
        if self._transform is not None:
            features = self._transform(features)
        tmp = self._meta[self._meta.id == id_]
        cat = tmp[tmp.image_numb == img_num].cat.iloc[0]  # topkek xD ))))
        label = self._cat_to_label[cat]
        return id_, img_num, features, label

    @staticmethod
    def _load_single_table(store):
        tmp = []
        keys = store.keys()
        for key in keys:
            tmp.append(store.select(key))
        table = pd.concat(tmp, ignore_index=True)
        return table

    @staticmethod
    def _load_tables(paths):
        tables = []
        for path in paths:
            print('Loading storage at {}'.format(path))
            storage = pd.HDFStore(path)
            tables.append(StackingDatasetServer._load_single_table(storage))
            print('Success, loaded table of shape {}.'.format(tables[0].shape))
        return tables

    def _select_id_and_img_num(self, index):
        assert len(self._tables) > 0
        ids = []
        img_nums = []
        for table in self._tables:
            ids.append(table.pr_id.iloc[index])
            img_nums.append(table.img_num.iloc[index])
        id_ = ids[0]
        img_num = img_nums[0]
        for id1, num1 in zip(ids, img_nums):
            assert id_ == id1
            assert img_num == num1
        return id_, img_num
