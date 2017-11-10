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

LOAD_WEIGHTS_FROM = config.DENSENET_DIR + '0_epoch_val.pth'
LOAD_OPTIM_FROM = None

TEST_BATCH_SIZE = 2048

NORM_MEAN = [0.49139968, 0.48215827, 0.44653124]
NORM_STD = [0.24703233, 0.24348505, 0.26158768]


def tta_transform(crop):
    return transforms.Compose([
        transforms.ToPILImage(),
        crop,
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
    ])


def tta_predict(model, inputs):
    res = list()
    transforms.FiveCrop(160) # ЭТОЙ ХУЙНИ ПОХОДУ ОПЯТЬ НЕТ


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
    model.train(False)
    for data in tqdm(dataloader,
                     total=np.ceil(test_size / float(TEST_BATCH_SIZE))):
        # get the inputs
        product_ids, image_numbers, inputs = data
        # wrap them in Variable
        assert torch.cuda.is_available()
        inputs = Variable(inputs.cuda(), volatile=True)
        outputs = model(inputs)
        proba = nn.functional.softmax(outputs.data).data
        _, preds = torch.max(proba, 1)


print()

time_elapsed = time.time() - since
print('Training complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))
print('Best val Acc: {:4f}'.format(best_acc))
# load best model weights
model.load_state_dict(best_model_wts)
return model

if __name__ == '__main__':
    train()
