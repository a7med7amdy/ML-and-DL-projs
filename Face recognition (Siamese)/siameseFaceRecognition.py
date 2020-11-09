import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.models as models
torch.set_printoptions(linewidth=120)
import skimage.io as io
import matplotlib.pyplot as plt
from torch.autograd import Variable
import cv2 as cv2
import matplotlib.pyplot as plt
import os
from collections import defaultdict
import random
from torchvision.transforms import ToTensor
from pathlib import Path
from PIL import Image
import sys
from sklearn.datasets import fetch_lfw_pairs
import PIL.ImageOps 
import glob
print(torch.__version__)
print(torchvision.__version__)

file1 = open('pairs.txt', 'r') 
Lines = file1.readlines() 
count = 0
dataset=[]
for line in Lines:
    if(line[0]=="#"):
        continue
    pair= line.strip().split("  ")
    img1= Image.open(pair[0])
    img1 = ToTensor()(img1)
    img1 = Variable(img1)
    img2= Image.open(pair[1])
    img2 = ToTensor()(img2)
    img2 = Variable(img2)
    dataset.append([img1,img2,torch.tensor(int(pair[2]))])
    
print(len(dataset))

random.shuffle(dataset)
random.shuffle(dataset)

# trainset, testset = torch.utils.data.random_split(dataset, [11040, 2760])
trainset= dataset[0:11040]
testset = dataset[11040::]

train_dataloader = torch.utils.data.DataLoader(trainset,num_workers=4,batch_size=32,shuffle=True)
test_dataloader = torch.utils.data.DataLoader(testset,num_workers=4,batch_size=32,shuffle=False)

class SiameseNet(nn.Module):

    """
    Defining Network
    """
    def __init__(self):
        super(SiameseNet, self).__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 5, stride=(1,1), padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d((2,2), stride=(2,2)),

            nn.Conv2d(64, 128, 5, stride=(1,1), padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d((2,2), stride=(2,2)),

            nn.Conv2d(128, 256, 3, stride=(1,1), padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.MaxPool2d((2,2), stride=(2,2)),

            nn.Conv2d(256, 512, 3, stride=(1,1), padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512)
        )

        self.fc1 = nn.Sequential(
            nn.Linear(73728, 1024),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1024)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )
        
    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    
    def forward_once(self, x):
        output = self.cnn(x)
        output = output.view(-1, self.num_flat_features(output))
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        output = self.fc2(abs(output1-output2))
        return output

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
Net = SiameseNet().to(device)
# optimizer = optim.Adam(Net.parameters(),lr=0.0001)
optimizer = optim.SGD(Net.parameters(), lr=0.0001, momentum=0.9)
# resnet = resnet.to(device)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=28, gamma=0.1)

def train():
    Net.train()
    for epoch in range(70):
#         scheduler.step()
        total_loss=0
        corrected=0
        for images1,images2,labels in train_dataloader:
            images1= images1.to(device)
            images2= images2.to(device)
            labels = labels.unsqueeze(1).to(device)
            optimizer.zero_grad()
            out = Net(images1,images2)
            labels = labels.type_as(out)
            loss = F.binary_cross_entropy(out, labels)
            out=out>=0.5
#             corrected+=(labels==out).sum().item()
            loss= loss.mean()
            total_loss+= loss.item()
            loss.backward()
            optimizer.step()
        print("epoch no = ",epoch," total_loss = ",total_loss)
#         print("corrected= ",corrected)
train()

@torch.no_grad()
def test():
    Net.eval()
    total_loss=0
    total=0
    corrected=0
    for images1,images2,labels in test_dataloader:
        images1= images1.to(device)
        images2= images2.to(device)
        labels = labels.unsqueeze(1).to(device)
        out = Net(images1,images2)
        labels = labels.type_as(out)
        loss = F.binary_cross_entropy(out, labels)
        loss= loss.mean()
        total+=labels.size(0)
        out=out>=0.5
        corrected+=(labels==out).sum().item()
        total_loss+= loss.item()
#         print(out," ------> ",labels)
    print("corrected= ",corrected)
    print("total_loss= ",total_loss)
    print("total= ",total)
test()

torch.save(Net.state_dict(), "siameseNew.pth")

f, axarr = plt.subplots(1,3)
img1= Image.open("test/amr1.jpg")
axarr[0].imshow(np.asarray(img1))
img1 = ToTensor()(img1)
img1 = Variable(img1)
img2= Image.open("test/amr2.jpg")
axarr[1].imshow(np.asarray(img2))
img2 = ToTensor()(img2)
img2 = Variable(img2)
img4= Image.open("img.jpg")
axarr[2].imshow(np.asarray(img4))
img4 = ToTensor()(img4)
img4 = Variable(img4)

Net.eval()
test=Net(img1.unsqueeze(0).to(device),img2.unsqueeze(0).to(device))
print(test>=0.5)

test2=Net(img2.unsqueeze(0).to(device),img4.unsqueeze(0).to(device))
print(test2>=0.5)

cnt=0
for i in trainset:
    if i[2]==1:
        cnt+=1
print(cnt)

cnt=0
for i in testset:
    if i[2]==1:
        cnt+=1
print(cnt)

print(len(testset))
