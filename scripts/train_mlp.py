from __future__ import print_function
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

import config
from mymodels.mlp import MLP

BATCH_SIZE = 500  # Number of samples in each batch
EPOCHS = 100  # Number of epochs to train the network


def next_batch(train=True):
    # A function to read the next batch of MNIST images and labels
    # Args:
    #   train: a boolean array, if True it will return the next train batch, otherwise the next test batch
    # Returns:
    #   batch_img: a pytorch Variable of size [batch_size, 748].
    #   batch_label: a pytorch Variable of size [batch_size, ].

    if train:
        batch_img, batch_label = mnist.train.next_batch(BATCH_SIZE)
    else:
        batch_img, batch_label = mnist.test.next_batch(BATCH_SIZE)

    batch_label = torch.from_numpy(
        batch_label).long()  # convert the numpy array into torch tensor
    batch_label = Variable(
        batch_label).cuda()  # create a torch variable and transfer it into GPU

    batch_img = torch.from_numpy(
        batch_img).float()  # convert the numpy array into torch tensor
    batch_img = Variable(
        batch_img).cuda()  # create a torch variable and transfer it into GPU
    return batch_img, batch_label


def main():
    # define the neural network (multilayer perceptron) and move the network into GPU
    net = MLP(config.CAT_COUNT * 2, config.CAT_COUNT)
    net.cuda()

    # calculate the number of batches per epoch
    batch_per_ep = mnist.train.num_examples // BATCH_SIZE

    # define the loss (criterion) and create an optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)

    for ep in range(EPOCHS):  # epochs loop
        for batch_n in range(batch_per_ep):  # batches loop
            features, labels = next_batch()

            # Reset gradients
            optimizer.zero_grad()

            # Forward pass
            output = net(features)
            loss = criterion(output, labels)

            # Backward pass and updates
            loss.backward()  # calculate the gradients (backpropagation)
            optimizer.step()  # update the weights

            if not batch_n % 10:
                print('epoch: {} - batch: {}/{} \n'.format(ep, batch_n,
                                                           batch_per_ep))
                print('loss: ', loss.data[0])

    # test the accuracy on a batch of test data
    features, labels = next_batch(train=False)
    print('\n \n Test accuracy: ', net.accuracy(features, labels))


if __name__ == '__main__':
    main()
