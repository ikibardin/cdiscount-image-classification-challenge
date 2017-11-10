import numpy as np
import torch
import torch.nn as nn
from pymongo import MongoClient
from torch.autograd import Variable
from torchvision import models
from tqdm import tqdm

import config
import loading
from label_to_cat import LABEL_TO_CAT

BEST_WEIGHTS = ''

USE_TTA = True

PRODUCTS_IN_BATCH = 4


def predict():
    model = models.inception_v3(pretrained=False, num_classes=config.CAT_COUNT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(LABEL_TO_CAT))
    assert torch.cuda.is_available()
    model = nn.DataParallel(model, device_ids=[0, 1]).cuda()
    model.load_state_dict(torch.load(BEST_WEIGHTS))
    print("Loaded weights from", BEST_WEIGHTS)
    predict_model(model, USE_TTA)


def get_test_generator():
    client = MongoClient(connect=False)
    db = client.cdiscount['test']
    products_loaded = 0
    imgs_out = list()
    ids = list()
    imgs_counts = list()
    for product in db.find():
        imgs_out += [
            loading._img_from_bytes(image_bytes['picture'])
            for image_bytes in product['imgs']
        ]
        ids.append(product['_id'])
        imgs_counts.append(len(imgs_out))
        products_loaded += 1
        if products_loaded == PRODUCTS_IN_BATCH:
            products_loaded = 0
            output = ids, imgs_counts, imgs_out
            ids = list()
            imgs_counts = list()
            imgs_out = list()
            yield output


def predict_model(model, use_tta):
    test_gen = get_test_generator()
    predictions = list()
    for ids_batch, imgs_counts, imgs_batch in tqdm(test_gen,
                                                   total=np.ceil(
                                                               config.TEST_PRODUCTS_COUNT / PRODUCTS_IN_BATCH)):
        inputs = Variable(imgs_batch.cuda())
        outputs = model(inputs.float())
        if use_tta:
            pass
        raise RuntimeError('Not implemented')
