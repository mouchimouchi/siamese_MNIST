def load_MNIST():
    trans = transforms.Compose(
        [transforms.Pad(2),
         transforms.ToTensor(),
         transforms.Normalize((0.5,), (0.5,))
         ])

    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                          download=True, transform=trans)

    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                         download=True, transform=trans)
    return trainset, testset


if __name__ == "__main__":

    import torch
    import torchvision

    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision.transforms as transforms
    import torch.optim as optim
    import numpy as np
    import random
    import argparse

    from make_pairs import *
    from model import *
    from train import *
    from test import *

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='CNN',
                        help='MLP or CNN')
    args = parser.parse_args()

    model_name = args.model

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    trainset, testset = load_MNIST()

    if model_name == "MLP":
        print("MLP is selected")
        model = siamese_MLP()
    else:
        model = siamese_CNN()
        print("CNN is selected")

    model.to(device)
    pair_trainset, pair_trainloader = make_pairloader(trainset, 5,
                                                      "train")
    # dataset, num_each (各クラス何枚ずつのデータを使うか),train or valid (違うとこからデータをとってくる。)
    pair_validset, pair_validloader = make_pairloader(trainset, 5, "valid")

    train(model, pair_trainloader, pair_validloader, 10)

    oneshot_accuracy = oneshot_mean(model, testset)