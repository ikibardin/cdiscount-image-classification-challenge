import pandas as pd
from pymongo import MongoClient
from tqdm import tqdm

import config


def build_test_ids():
    print('Building test ids...')
    client = MongoClient(connect=False)
    db = client.cdiscount['test']
    data = list()
    for product in tqdm(db.find(), total=config.TEST_PRODUCTS_COUNT):
        assert 1 <= len(product['imgs']) <= 4
        for i in range(len(product['imgs'])):
            data.append((product['_id'], i))
    df = pd.DataFrame(data=data, columns=['id', 'image_numb'])
    df.to_csv('test_ids.csv')
    print('Generated dataframe with shape', df.shape)
    print('Done.')


if __name__ == '__main__':
    build_test_ids()
