from __future__ import print_function, division

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import pandas as pd
from tqdm import tqdm

import config
import loading
from mymodels.densenet import densenet201
import tta_predict
from probs_saver import ProbStore

LOAD_WEIGHTS_FROM = config.DENSENET_DIR + '27_epoch_val.pth'

TEST_BATCH_SIZE = 256

NORM_MEAN = [0.49139968, 0.48215827, 0.44653124]
NORM_STD = [0.24703233, 0.24348505, 0.26158768]


def train():
    # ids_test = pd.read_csv(config.TEST_IDS_PATH)
    ids_valid = pd.read_csv(config.ARTUR_VALID_PATH)
    ids_test = ids_valid
    print('Predicting on {} samples.'.format(ids_test.shape[0]))

    test_dataset = loading.CdiscountDatasetPandas(
        img_ids_df=ids_test,
        mode='valid',
        transform=tta_predict.tta_transform(NORM_MEAN, NORM_STD))

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=TEST_BATCH_SIZE,
        shuffle=False,
        num_workers=1
    )
    # print(len(test_loader))
    model = densenet201(pretrained=True, num_classes=config.CAT_COUNT)
    assert torch.cuda.is_available()
    model = nn.DataParallel(model, device_ids=[0, 1]).cuda()
    assert LOAD_WEIGHTS_FROM is not None
    model.load_state_dict(torch.load(LOAD_WEIGHTS_FROM))
    model.cuda()
    predict(model, dataloader=test_loader, test_size=len(test_loader))


def predict(model, dataloader, test_size):
    columns1 = ['pr_id', 'img_num']
    columns2 = []
    for i in range(1, config.CAT_COUNT + 1):
        columns2.append(str(i))
    storage = ProbStore(path='../input/densenet201_valid.h5')
    model.train(False)
    for data in tqdm(dataloader, total=test_size):
        # get the inputs
        product_ids, image_numbers, inputs = data
        # wrap them in Variable
        assert torch.cuda.is_available()

        inputs = Variable(inputs.cuda(), volatile=True)
        bs, ncrops, c, h, w = inputs.size()
        # assert bs == TEST_BATCH_SIZE and ncrops == 10
        outputs = model(inputs.view(-1, c, h, w))
        proba = nn.functional.softmax(outputs.data).cpu()

        two_cols = np.array(list(zip(product_ids, image_numbers)))

        df1 = pd.DataFrame(two_cols, dtype=np.int32, columns=columns1,
                           index=None)
        df2 = pd.DataFrame(
            proba.data.view(-1, 10, config.CAT_COUNT).sum(
                dim=1).numpy().astype('float16'),
            columns=columns2, index=None, dtype=np.float16)
        df = pd.concat([df1, df2], axis=1)
        storage.saveProbs(df)


if __name__ == '__main__':
    train()
