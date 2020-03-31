import torch
import torchvision

import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.optim as optim

import numpy as np
import random

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train(model,train_loader,valid_loader,num_epoch):
  criterion = nn.MSELoss()
  optimizer = optim.SGD(model.parameters(),lr=0.001,momentum=0.9,weight_decay = 0.01)
  for i in range(num_epoch):
    losses = []
    model.train()
    for data,label in train_loader:
      data = data.to(device)
      label = label.to(device)

      img1 = data[:,0]
      img2 = data[:,1]

      output = model(img1,img2)
      label = label.view(-1,1)
      #print(label)
      loss = criterion(output,label)

      losses.append(loss.item())


      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      model.eval()
      valid_losses = []
    with torch.no_grad():
      for data,label in valid_loader:
        data = data.to(device)
        label = label.to(device)

        img1 = data[:,0]
        img2 = data[:,1]

        valid_output = model(img1,img2)
        label = label.view(-1,1)
        valid_loss = criterion(valid_output,label)
        valid_losses.append(valid_loss.item())

    print("epoch: {} loss:{}".format(i + 1,np.array(losses).mean()) + " valid_loss:{}".format(np.array(valid_losses).mean()))
  print("training finished")
