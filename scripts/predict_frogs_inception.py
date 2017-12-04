from __future__ import print_function, division

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import pandas as pd
from tqdm import tqdm

import config
import loading
from mymodels.inception_v3 import Inception3
import tta_predict
from probs_saver import ProbStore

LOAD_WEIGHTS_FROM = '../frogs-code/release/trained_models/LB=0.69565_inc3_00075000_model.pth'

TEST_BATCH_SIZE = 256

NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]


def train():
    ids_test = pd.read_csv(config.TEST_IDS_PATH)
    print('Predicting on {} samples.'.format(ids_test.shape[0]))

    test_dataset = loading.CdiscountDatasetPandas(
        img_ids_df=ids_test,
        mode='test',
        transform=tta_predict.frogs_transform())

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=TEST_BATCH_SIZE,
        shuffle=False,
        num_workers=1
    )
    assert torch.cuda.is_available()
    # print(len(test_loader))
    model = Inception3((3, 180, 180), num_classes=config.CAT_COUNT)
    assert LOAD_WEIGHTS_FROM is not None
    model.load_pretrain_pytorch_file(LOAD_WEIGHTS_FROM, skip=[])
    model = nn.DataParallel(model, device_ids=[0, 1]).cuda()
    model.cuda()
    predict(model, dataloader=test_loader, test_size=len(test_loader))


def predict(model, dataloader, test_size):
    columns1 = ['pr_id', 'img_num']
    columns2 = []
    for i in range(1, config.CAT_COUNT + 1):
        columns2.append(str(i))
    storage = ProbStore(path='../input/inc3_test_xDDD.h5')
    model.train(False)
    for data in tqdm(dataloader, total=test_size):
        # get the inputs
        product_ids, image_numbers, inputs = data
        # wrap them in Variable
        assert torch.cuda.is_available()

        inputs = Variable(inputs.cuda(), volatile=True)
        bs, c, h, w = inputs.size()
        # assert bs == TEST_BATCH_SIZE and ncrops == 10
        outputs = model(inputs)
        proba = nn.functional.softmax(outputs.data).cpu()

        two_cols = np.array(list(zip(product_ids, image_numbers)),
                            dtype=np.int32)

        df1 = pd.DataFrame(two_cols, dtype=np.int32, columns=columns1,
                           index=None)
        df2 = pd.DataFrame(proba.data.numpy().astype('float16'),
                           columns=columns2, index=None, dtype=np.float16)
        df = pd.concat([df1, df2], axis=1)
        storage.saveProbs(df)


if __name__ == '__main__':
    train()
