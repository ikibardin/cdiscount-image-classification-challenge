from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import config
import loading
from label_to_cat import LABEL_TO_CAT

BATCH_SIZE = 32
EPOCHS = 3
ITERS_PER_EPOCH = 10000


def train():
    all_imgs_ids = loading.load_all_train_imgs_ids()
    ids_train, ids_valid = train_test_split(all_imgs_ids, test_size=0.001,
                                            random_state=0)
    train_dataset = loading.CdiscountDataset(ids_train,
                                             'train',
                                             transform=transforms.Compose(
                                                 [loading.resize,
                                                  transforms.ToTensor(),
                                                  transforms.Normalize(
                                                      [0.485, 0.456, 0.406],
                                                      [0.229, 0.224, 0.225])
                                                  ]))

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4
    )
    valid_dataset = loading.CdiscountDataset(ids_valid, 'train',
                                             transform=transforms.Compose(
                                                 [loading.resize,
                                                  transforms.ToTensor(),
                                                  transforms.Normalize(
                                                      [0.485, 0.456, 0.406],
                                                      [0.229, 0.224, 0.225])
                                                  ]))
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4
    )
    dataloaders = {
        'train': train_loader,
        'val': valid_loader
    }
    dataset_sizes = {
        'train': len(train_dataset),
        'val': len(valid_dataset)
    }

    model = models.inception_v3(pretrained=True)
    model.aux_logits = False
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(LABEL_TO_CAT))
    assert torch.cuda.is_available()
    model.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    model = train_model(model, dataloaders,
                        dataset_sizes, criterion,
                        optimizer, exp_lr_scheduler, EPOCHS)


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
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            if phase == 'train':
                iternum = 0
            total_iters = ITERS_PER_EPOCH if phase == 'train' else \
                int(np.ceil(dataset_sizes[phase] / float(BATCH_SIZE)))

            for data in tqdm(dataloaders[phase], total=total_iters):
                # get the inputs
                if phase == 'train' and iternum == ITERS_PER_EPOCH:
                    break
                inputs, labels = data

                # wrap them in Variable
                assert torch.cuda.is_available()
                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda())

                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                outputs = model(inputs.float())
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)
                # print("LOSS ", loss.data)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.data[0]
                running_corrects += torch.sum(preds == labels.data)
                if phase == 'train':
                    iternum += 1
            if phase == 'test':
                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects / dataset_sizes[phase]
            else:
                epoch_loss = running_loss / (iternum * BATCH_SIZE)
                epoch_acc = running_corrects / (iternum * BATCH_SIZE)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
                torch.save(best_model_wts,
                           config.INCEPTION_V3_DIR + '{}_epoch.pth'.format(
                               epoch))
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
