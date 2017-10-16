import pickle
from pymongo import MongoClient
from torch.utils.data import Dataset
from tqdm import tqdm

import config


def build_ids(mode):
    assert mode in ('train', 'test')
    res = list()
    client = MongoClient(connect=False)
    db = client.cdiscount[mode]
    for product in tqdm(db.find(), total=config.PRODUCT_COUNT):
        assert 1 <= len(product['imgs']) <= 4
        for i in range(len(product['imgs'])):
            res.append((product['_id'], i))
    with open(config.IMG_IDS_PATH, 'wb') as f:
        pickle.dump(res, f)
    print('Done.')


if __name__ == '__main__':
    build_ids('train')
