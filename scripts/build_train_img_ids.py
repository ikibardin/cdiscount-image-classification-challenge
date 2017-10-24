import pandas as pd
import pickle
from pymongo import MongoClient
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


def build_separator_ids():
    print('Building separator images ids.')
    df = pd.read_csv(config.SEPARATOR_IMG_IDS_PATH_CSV)
    res = list()
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        res.append((row['_id'], row['img']))
    with open(config.SEPARATOR_IMG_IDS_PATH_PICKLE, 'wb') as f:
        pickle.dump(res, f)


if __name__ == '__main__':
    # build_separator_ids()
    build_ids('train', with_cat_id=True)
