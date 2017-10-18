import pickle
from pymongo import MongoClient
from torch.utils.data import Dataset
from tqdm import tqdm

import config


def build_ids(mode, with_cat_id=False):
    assert mode in ('train', 'test')
    res = list()
    client = MongoClient(connect=False)
    db = client.cdiscount[mode]
    for product in tqdm(db.find(), total=config.PRODUCT_COUNT):
        assert 1 <= len(product['imgs']) <= 4
        for i in range(len(product['imgs'])):
            if with_cat_id:
                res.append((product['_id'], i, product['category_id']))
            else:
                res.append((product['_id'], i))
    if with_cat_id:
        path = config.IMG_IDS_WITH_CAT_PATH
    else:
        path = config.IMG_IDS_PATH
    with open(path, 'wb') as f:
        pickle.dump(res, f)
    print('Done.')


if __name__ == '__main__':
    build_ids('train', with_cat_id=True)
