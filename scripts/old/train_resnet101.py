from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, transforms
import time
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import gc
import random
from PIL import Image

import config
import loading
from label_to_cat import LABEL_TO_CAT
from mymodels.resnet import resnet101

BATCH_SIZE = 128
EPOCHS = 50
VALID_SIZE = 851293

PHASE_TRAIN = 'train'
PHASE_VAL = 'val'

INITIAL_LR = 0.001


class MyRandomVerticalFlip(object):
    """Vertically flip the given PIL Image randomly with a
    probability of 0.5."""

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be flipped.
        Returns:
            PIL Image: Randomly flipped image.
        """
        if random.random() < 0.5:
            return img.transpose(Image.FLIP_TOP_BOTTOM)
        return img


def train():
    all_imgs_ids = loading.load_all_train_imgs_ids()
    assert config.TRAIN_SIZE == len(all_imgs_ids)
    ids_train, ids_valid = train_test_split(all_imgs_ids,
                                            test_size=VALID_SIZE,
                                            random_state=0)
    print("Training on {} samples, validating on {} samples.".format(
        len(ids_train),
        len(ids_valid)))
    train_dataset = loading.CdiscountDatasetFromPickledList(ids_train,
                                                            PHASE_TRAIN,
                                                            transform=transforms.Compose(
                                                 [transforms.ToPILImage(),
                                                  transforms.RandomCrop(160),
                                                  transforms.RandomHorizontalFlip(),
                                                  MyRandomVerticalFlip(),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize(
                                                      [0.485, 0.456, 0.406],
                                                      [0.229, 0.224, 0.225])
                                                  ]),
                                                            big_cats=False)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=1
    )
    valid_dataset = loading.CdiscountDatasetFromPickledList(ids_valid, PHASE_TRAIN,
                                                            transform=transforms.Compose(
                                                 [transforms.ToPILImage(),
                                                  transforms.RandomCrop(160),
                                                  # transforms.RandomHorizontalFlip(),
                                                  # transforms.RandomVerticalFlip(),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize(
                                                      [0.485, 0.456, 0.406],
                                                      [0.229, 0.224, 0.225])
                                                  ]),
                                                            big_cats=False)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
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

    model = resnet101(pretrained=True, num_classes=config.CAT_COUNT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(LABEL_TO_CAT))
    assert torch.cuda.is_available()
    model = nn.DataParallel(model, device_ids=[0, 1]).cuda()
    criterion = nn.CrossEntropyLoss()
    criterion.cuda()
    optimizer = optim.SGD(model.parameters(), lr=INITIAL_LR, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    model = train_model(model, dataloaders,
                        dataset_sizes, criterion,
                        optimizer, exp_lr_scheduler, EPOCHS)


def preds_from_outputs(outputs):
    _, preds = torch.max(
        outputs.data, 1
    )
    return preds


def train_model(model, dataloaders, dataset_sizes,
                criterion, optimizer, scheduler,
                num_epochs):
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 20)

        # Each epoch has a training and validation phase
        for phase in [PHASE_TRAIN, PHASE_VAL]:
            if phase == PHASE_TRAIN:
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for data in tqdm(dataloaders[phase],
                             total=np.ceil(dataset_sizes[phase] / BATCH_SIZE)):
                # get the inputs
                inputs, labels = data
                # wrap them in Variable
                assert torch.cuda.is_available()
                if phase == PHASE_TRAIN:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda(async=True))
                else:
                    inputs = Variable(inputs.cuda(), volatile=True)
                    labels = Variable(inputs.cuda(async=True, volatile=True)
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                outputs = model(inputs)
                preds = preds_from_outputs(outputs)
                loss = criterion(outputs, labels)
                # backward + optimize only if in training phase
                if phase == PHASE_TRAIN:
                    loss.backward()
                    optimizer.step()
                # statistics
                running_loss += loss.data[0]
                running_corrects += torch.sum(preds == labels.data)

                del inputs
                del labels
                gc.collect()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc)
            )
            torch.save(model.state_dict(),
                       config.RESNET101_DIR + '{}_epoch_{}.pth'.format(
                           epoch, phase))
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
