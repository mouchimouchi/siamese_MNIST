import torch
import torchvision

import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.optim as optim

import numpy as np
import random



class siamese_MLP(nn.Module):

    def __init__(self):
        super(siamese_MLP, self).__init__()
        self.fc1 = nn.Linear(1024, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 1, bias=False)

        self.fc4 = nn.Linear(1024, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        x1 = x1.view(-1, 1, 1024)
        x1 = F.relu(self.fc1(x1))
        x1 = F.relu(self.fc2(x1))

        x2 = x2.view(-1, 1, 1024)
        x2 = F.relu(self.fc1(x2))
        x2 = F.relu(self.fc2(x2))

        x1 = x1.view(-1, 1024)
        x2 = x2.view(-1, 1024)

        distance = F.l1_loss(x1, x2, reduction='none')
        out = self.fc4(distance)
        out = self.sigmoid(out)

        return out

class siamese_CNN(nn.Module):

  def __init__(self):
    super(siamese_CNN,self).__init__()
    self.conv1 = nn.Conv2d(1,64,3) #64@30*30 論文では、元画像105でカーネル10なので、元画像32に対しては、約1/3の3にした。
    self.pool1 = nn.MaxPool2d(2,stride=2) #64@15*15

    self.conv2 = nn.Conv2d(64,128,2) #128@14*14
    self.pool2 = nn.MaxPool2d(2,stride=2) #128@7*7

    self.conv3 = nn.Conv2d(128,256,2) #256@6*6
    #self.pool3 = nn.MaxPool2d(2,stride=2) #以下、小さすぎるのでなくす

    #self.conv4 = nn.Conv2d(128,256,4)

    self.fc = nn.Linear(256*6*6,1,bias=False)
    self.sigmoid = nn.Sigmoid()

  def forward_single(self,x):
    x = x.view(-1,1,32,32)
    x = F.relu(self.conv1(x))
    x = self.pool1(x)
    x = F.relu(self.conv2(x))
    x = self.pool2(x)
    x = F.relu(self.conv3(x))
    x = x.view(-1,256*6*6)

    return x

  def forward(self,x1,x2):
    x1 = self.forward_single(x1)
    x2 = self.forward_single(x2)

    distance = F.l1_loss(x1,x2,reduction='none')
    out = self.fc(distance)
    out = self.sigmoid(out)

    return out
