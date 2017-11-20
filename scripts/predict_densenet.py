from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import pandas as pd
from torchvision import transforms
import time
from tqdm import tqdm

import config
import loading
from mymodels.densenet import densenet201
import tta_predict
from probs_saver import ProbStore

LOAD_WEIGHTS_FROM = config.DENSENET_DIR + '0_epoch_val.pth'
LOAD_OPTIM_FROM = None

TEST_BATCH_SIZE = 2048

NORM_MEAN = [0.49139968, 0.48215827, 0.44653124]
NORM_STD = [0.24703233, 0.24348505, 0.26158768]


def train():
    ids_test = pd.read_csv(config.TEST_IDS_PATH)
    print('Predicting on {} samples.'.format(ids_test.shape[0]))

    test_dataset = loading.CdiscountDatasetPandas(
        img_ids_df=ids_test,
        mode='test',
        transform=None)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=TEST_BATCH_SIZE,
        shuffle=True,
        num_workers=7
    )

    model = densenet201(pretrained=True, num_classes=config.CAT_COUNT)
    assert torch.cuda.is_available()
    model = nn.DataParallel(model, device_ids=[0, 1]).cuda()
    assert LOAD_WEIGHTS_FROM is not None
    model.load_state_dict(torch.load(LOAD_WEIGHTS_FROM))
    model.cuda()

    predict(model, dataloader=test_loader, test_size=len(test_loader))


def predict(model, dataloader, test_size):
    columns = ['pr_id', 'img_num']
    for i in range(1, config.CAT_COUNT + 1):
        columns.append(str(i))
    storage = ProbStore()
    model.train(False)
    transform = tta_predict.tta_transform(NORM_MEAN, NORM_STD)
    for data in tqdm(dataloader,
                     total=int(np.ceil(test_size / float(TEST_BATCH_SIZE)))):
        # get the inputs
        product_ids, image_numbers, inputs = data
        # wrap them in Variable
        assert torch.cuda.is_available()

        inputs = [transform(img) for img in inputs]
        inputs = Variable(inputs.cuda(), volatile=True)
        bs, ncrops, c, h, w = inputs.size()
        assert bs == TEST_BATCH_SIZE and ncrops == 10
        outputs = model(inputs.view(-1, c, h, w))
        proba = nn.functional.softmax(outputs.data)

        new_ids = []
        for x in product_ids:
            new_ids += [x] * 10
        new_numbers = []
        for x in image_numbers:
            new_numbers += [x] * 10

        df = pd.DataFrame(data=np.hstack(
            [new_ids, new_numbers, proba.numpy()]
        ), columns=columns, index=None)
        storage.saveProbs(df)


if __name__ == '__main__':
    train()
