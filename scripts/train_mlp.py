from __future__ import print_function
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from torchvision import transforms
from sklearn.model_selection import train_test_split

import config
from mymodels.mlp import MLP
from loading import StackingDataset

BATCH_SIZE = 512  # Number of samples in each batch

INITIAL_EPOCH = 0
EPOCHS = 1000  # Number of epochs to train the network

PATHS = ['../input/resnet50_valid_corr.h5',
         '../input/dense_valid_corr.h5',
         '../input/resnet101_valid.h5']

PHASE_TRAIN = 'train'
PHASE_VAL = 'val'


def make_loader(ids, dataset):
    sampler = torch.utils.data.sampler.SubsetRandomSampler(ids)
    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        sampler=sampler,
        num_workers=0
    )
    return loader


def totensor(features):
    return torch.from_numpy(features).float()


def main():
    print('Loading dataset...')
    dataset = StackingDataset(paths=PATHS, transform=totensor,
                              meta_path='../input/val.csv')
    print('Loaded dataset with length {}'.format(len(dataset)))
    all_ids = np.arange(0, len(dataset), 1)
    ids_train, ids_valid = train_test_split(all_ids, test_size=0.2,
                                            random_state=0)
    print('Training on {} samples, validating on {} samples'.format(
        len(ids_train),
        len(ids_valid)))

    train_loader = make_loader(ids_train, dataset)
    valid_loader = make_loader(ids_valid, dataset)

    loaders = {
        PHASE_TRAIN: train_loader,
        PHASE_VAL: valid_loader
    }

    dataset_sizes = {
        PHASE_TRAIN: len(ids_train),
        PHASE_VAL: len(ids_valid)
    }

    model = MLP(config.CAT_COUNT * len(PATHS), config.CAT_COUNT)
    model.cuda()

    # define the loss (criterion) and create an optimizer
    criterion = nn.CrossEntropyLoss(size_average=False)
    criterion.cuda()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    for epoch in range(EPOCHS):  # epochs loop
        print('Epoch {}/{}'.format(epoch, EPOCHS - 1))
        print('-' * 20)  # Each epoch has a training and validation phase
        for phase in [PHASE_TRAIN, PHASE_VAL]:
            if phase == PHASE_TRAIN:
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for data in tqdm(loaders[phase]):
                # get the inputs
                ids, img_nums, inputs, labels = data
                # wrap them in Variable
                assert torch.cuda.is_available()
                if phase == PHASE_TRAIN:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs = Variable(inputs.cuda(), volatile=True)
                    labels = Variable(labels.cuda(),
                                      volatile=True)
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                outputs = model(inputs)
                proba = nn.functional.softmax(outputs.data).data
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
                       config.MLP_DIR + '{}_epoch_{}.pth'.format(
                           epoch, phase))
            torch.save(optimizer.state_dict(),
                       config.MLP_DIR + 'optim.pth')


if __name__ == '__main__':
    main()
