from __future__ import print_function, division

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import pandas as pd
from tqdm import tqdm

import config
import loading
from mymodels.resnet import resnet50
import tta_predict
from probs_saver import ProbStore

LOAD_WEIGHTS_FROM = config.RESNET50_DIR + '23_epoch_val.pth'

TEST_BATCH_SIZE = 2048

NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]


def train():
    ids_test = pd.read_csv(config.TEST_IDS_PATH)
    ids_test = ids_test[:TEST_BATCH_SIZE * 5]
    print('Predicting on {} samples.'.format(ids_test.shape[0]))

    test_dataset = loading.CdiscountDatasetPandas(
        img_ids_df=ids_test,
        mode='test',
        transform=tta_predict.tta_transform(NORM_MEAN, NORM_STD))

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=TEST_BATCH_SIZE,
        shuffle=True,
        num_workers=7
    )

    model = resnet50(pretrained=True, num_classes=config.CAT_COUNT)
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
    storage = ProbStore(path='../input/predict_probs_resnet50.h5')
    model.train(False)
    for data in tqdm(dataloader,
                     total=int(np.ceil(test_size / float(TEST_BATCH_SIZE)))):
        # get the inputs
        product_ids, image_numbers, inputs = data
        # wrap them in Variable
        assert torch.cuda.is_available()

        # inputs = [transform(img) for img in inputs]
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
