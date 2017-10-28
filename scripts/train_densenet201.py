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

LOAD_WEIGHTS_FROM = None
LOAD_OPTIM_FROM = None

INITIAL_EPOCH = 0

BATCH_SIZE = 256
VAL_BATCH_SIZE = 128

EPOCHS = 100

PHASE_TRAIN = 'train'
PHASE_VAL = 'val'


def train():
    ids_train = pd.read_csv(config.ARTUR_TRAIN_PATH)
    ids_valid = pd.read_csv(config.ARTUR_VALID_PATH)
    print("Training on {} samples, validating on {} samples.".format(
        ids_train.shape[0], ids_valid.shape[0]))
    norm_mean = [0.49139968, 0.48215827, 0.44653124]
    norm_std = [0.24703233, 0.24348505, 0.26158768]
    train_dataset = loading.CdiscountDatasetPandas(
        ids_train,
        PHASE_TRAIN,
        transform=transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop(160),
            transforms.ToTensor(),
            transforms.Normalize(mean=norm_mean, std=norm_std)
        ]),
        big_cats=False)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=1
    )
    valid_dataset = loading.CdiscountDatasetPandas(
        ids_valid,
        PHASE_TRAIN,
        transform=transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop(160),
            transforms.ToTensor(),
            transforms.Normalize(mean=norm_mean, std=norm_std)
        ]),
        big_cats=False)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=VAL_BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )
    dataloaders = {
        PHASE_TRAIN: train_loader,
        PHASE_VAL: valid_loader
    }
    dataset_sizes = {
        PHASE_TRAIN: len(train_dataset),
        PHASE_VAL: len(valid_dataset)
    }

    model = densenet201(pretrained=True, num_classes=config.CAT_COUNT)
    assert torch.cuda.is_available()
    model = nn.DataParallel(model, device_ids=[0, 1]).cuda()
    if LOAD_WEIGHTS_FROM is not None:
        model.load_state_dict(torch.load(LOAD_WEIGHTS_FROM))
    model.cuda()
    criterion = nn.CrossEntropyLoss()
    criterion.cuda()

    # Freeze layers
    for param in model.parameters():
        param.requires_grad = False
    for param in model.module.classifier.parameters():
        param.requires_grad = True

    optimizer = optim.Adam(model.module.classifier.parameters(), lr=0.0001)
    if LOAD_OPTIM_FROM is not None:
        optimizer.load_state_dict(torch.load(LOAD_OPTIM_FROM))

    model = train_model(model, dataloaders,
                        dataset_sizes, criterion,
                        optimizer, EPOCHS)


def train_model(model, dataloaders, dataset_sizes,
                criterion, optimizer, num_epochs):
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(INITIAL_EPOCH, num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 20)

        # Each epoch has a training and validation phase
        for phase in [PHASE_VAL, PHASE_TRAIN]:
            if phase == PHASE_TRAIN:
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            if phase == PHASE_TRAIN:
                phase_batch_size = BATCH_SIZE
            else:
                phase_batch_size = VAL_BATCH_SIZE
            for data in tqdm(dataloaders[phase],
                             total=np.ceil(dataset_sizes[
                                               phase] / phase_batch_size)):
                # get the inputs
                inputs, labels = data
                # wrap them in Variable
                assert torch.cuda.is_available()
                if phase == PHASE_TRAIN:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda(async=True))
                else:
                    inputs = Variable(inputs.cuda(), volatile=True)
                    labels = Variable(labels.cuda(async=True), volatile=True)
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                outputs = model(inputs)
                proba = nn.functional.softmax(outputs.data)
                _, preds = torch.max(proba, 1)
                loss = criterion(outputs, labels)
                # backward + optimize only if in training phase
                if phase == PHASE_TRAIN:
                    loss.backward()
                    optimizer.step()
                # statistics
                running_loss += loss.data[0]
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc)
            )
            torch.save(model.state_dict(),
                       config.DENSENET_DIR + '{}_epoch_{}.pth'.format(
                           epoch, phase))
            torch.save(optimizer.state_dict(),
                       config.DENSENET_DIR + 'optim.pth')
            # deep copy the model
            if phase == PHASE_VAL and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
                print('Best weights updated!')

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
