import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
torch.set_printoptions(linewidth=120)
import skimage.io as io
import cv2 as cv2
import os
from torch.utils.tensorboard import SummaryWriter
print(torch.__version__)
print(torchvision.__version__)
from torch.autograd import Variable

def prepare_data(batch_size,path):
    data_dir = path
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(device)

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(100),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(100),
            transforms.CenterCrop(100),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
         'test': transforms.Compose([
            transforms.Resize(100),
            transforms.CenterCrop(100),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }


    folders=os.listdir(os.path.join(data_dir, 'train'))
    data_set = {x: torchvision.datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x])
                  for x in ['train', 'val','test']}
    train_loader = {x: torch.utils.data.DataLoader(data_set[x], batch_size=batch_size,
                                                 shuffle=True, num_workers=4)
                  for x in ['train', 'val','test']}
    return folders,data_set,train_loader,device

def get_num_correct(preds,labels):
    return preds.argmax(dim=1).eq(labels).sum().item()

#class focal loss to calculate the loss depending on the hard examples
#focal loss class is for multi-label classification
def one_hot(index, classes,device):
    size = index.size() + (classes,)
    view = index.size() + (1,)

    mask = torch.Tensor(*size).fill_(0)
    mask=mask.to(device)
    index = index.view(*view)
    index=index.to(device)
    ones = 1.

    if isinstance(index, Variable):
        ones = Variable(torch.Tensor(index.size()).fill_(1))
        ones=ones.to(device)
        mask = Variable(mask, volatile=index.volatile)

    return mask.scatter_(1, index, ones)
class FocalLoss(nn.Module):

    def __init__(self, gamma=0, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps

    def forward(self, input, target,device):
        y = one_hot(target, input.size(-1),device)
        logit = F.softmax(input, dim=-1)
        logit = logit.clamp(self.eps, 1. - self.eps)

        loss = -1 * y * torch.log(logit) # cross entropy
        loss = loss * (1 - logit) ** self.gamma # focal loss

        return loss.sum()

def get_resnet(n,device):
    resnet50 = models.resnet50(pretrained=True)
    for param in resnet50.parameters():
        param.require_grad = False

    num_ftrs = resnet50.fc.in_features
    resnet50.fc = nn.Linear(num_ftrs, n)
    resnet50 = resnet50.to(device)
    return resnet50

def model(resnet50,lr,n,device,train_loader,batch_size,epoch_num,tb):
    optimizer= optim.Adam(resnet50.parameters(),lr=lr)
    focal_loss_multilabel=FocalLoss()
    for epoch in range (epoch_num):
        total_loss=0
        total_correct=0
        #images, labels represent current batch
        for images,labels in train_loader['train']:
            images = images.to(device)
            labels = labels.to(device)
#             grid=torchvision.utils.make_grid(images)
#             tb.add_images('images',grid, dataformats='CHW')
#             tb.add_graph(resnet50,images)
#             tb.close()
            preds=resnet50(images)
            loss=None
            if(n>=3):
                #if you want to calculate it using the normal cross entropy
#                 loss = F.cross_entropy(preds,labels)
                #if you want to calculate it using the focal loss
                loss=focal_loss_multilabel.forward(preds,labels,device)
            else:
                #if you want to calculate it using the normal cross entropy
#                 loss = F.binary_cross_entropy(preds,labels)
                #if you want to calculate it using the focal loss
                BCE_loss = F.binary_cross_entropy_with_logits(preds, labels, reduction='none')
                pt = torch.exp(-BCE_loss) # prevents nans when probability 0
                Focal_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
                loss= Focal_loss.mean()
                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss+=loss.item()
            total_correct+=get_num_correct(preds,labels)
            print ("epoch: ",epoch," total_correct: ",total_correct," total_loss: ",total_loss)
        tb.add_scalar('loss',total_loss,epoch)
        tb.add_scalar('number correct',total_correct,epoch)
        tb.add_scalar('accuracy',total_correct/(len(train_loader['train'])*batch_size),epoch)
    return total_correct,total_loss

def print_train_accuracy(total_correct,train_loader,batch_size):
    print("train accuracy: ",total_correct/(len(train_loader['train'])*batch_size))

@torch.no_grad()
def validation(resnet50,device,train_loader,n,tb,batch_size):
    total_loss=0
    total_correct=0
    focal_loss_multilabel=FocalLoss()
    for images,labels in train_loader['val']:
        images = images.to(device)
        labels = labels.to(device)
        preds=resnet50(images)
        loss=None
        
        if(n>=3):
            #if you want to calculate it using the normal cross entropy
#                 loss = F.cross_entropy(preds,labels)
            #if you want to calculate it using the focal loss
            loss=focal_loss_multilabel.forward(preds,labels,device)
        else:
            #if you want to calculate it using the normal cross entropy
#                 loss = F.binary_cross_entropy(preds,labels)
            #if you want to calculate it using the focal loss
            BCE_loss = F.binary_cross_entropy_with_logits(preds, labels, reduction='none')
            pt = torch.exp(-BCE_loss) # prevents nans when probability 0
            Focal_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
            loss= Focal_loss.mean()
            
        total_loss+=loss.item()
        total_correct+=get_num_correct(preds,labels)
        print ("total_correct: ",total_correct," total_loss: ",total_loss)
    tb.add_scalar('loss',total_loss,batch_size)
    tb.add_scalar('number correct',total_correct,batch_size)
    tb.add_scalar('accuracy',total_correct/(len(train_loader['train'])*batch_size),batch_size)
    return total_correct,total_loss

def print_validation_accuracy(total_correct,train_loader,batch_size):
    print("validation accuracy: ",total_correct/(len(train_loader['val'])*batch_size))


@torch.no_grad()
def testing(resnet50,device,train_loader,n):
    total_loss=0
    total_correct=0
    focal_loss_multilabel=FocalLoss()
    for images,labels in train_loader['test']:
        images = images.to(device)
        labels = labels.to(device)
        preds=resnet50(images)
        loss=None
        if(n>=3):
            #if you want to calculate it using the normal cross entropy
#                 loss = F.cross_entropy(preds,labels)
            #if you want to calculate it using the focal loss
            loss=focal_loss_multilabel.forward(preds,labels,device)
        else:
            #if you want to calculate it using the normal cross entropy
#                 loss = F.binary_cross_entropy(preds,labels)
            #if you want to calculate it using the focal loss
            BCE_loss = F.binary_cross_entropy_with_logits(preds, labels, reduction='none')
            pt = torch.exp(-BCE_loss) # prevents nans when probability 0
            Focal_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
            loss= Focal_loss.mean()
        total_loss+=loss.item()
        total_correct+=get_num_correct(preds,labels)
        print ("total_correct: ",total_correct," total_loss: ",total_loss)
    return total_correct,total_loss

def print_testing_accuracy(total_correct,train_loader,batch_size):
    print("testing_accuracy: ",total_correct/(len(train_loader['test'])*batch_size))

def main():
    #     batch_size_list=[32,64,128]
#     lr_list=[0.0001,0.00001,0.000001]
#     epoch_num_list=[10,17,23]
#     for batch_size in batch_size_list :
#         for lr in lr_list :
#             for epoch_num in epoch_num_list:
                batch_size=64
                lr=0.00001
                epoch_num=23
                folder,data_set,train_loader,device=prepare_data(batch_size,"/home/ahmed/intern work/image classification/animal-image-datasetdog-cat-and-panda")
                print(len(folder))
                print("no of data = ",len(train_loader['train'])*batch_size)
                n=len(folder)
                tb=SummaryWriter()
                resnet50=get_resnet(n,device)
                total_correct,total_loss=model(resnet50,lr,n,device,train_loader,batch_size,epoch_num,tb)
                print_train_accuracy(total_correct,train_loader,batch_size)
                total_correct,total_loss=validation(resnet50,device,train_loader,n,tb,batch_size)
                print_validation_accuracy(total_correct,train_loader,batch_size)
                total_correct,total_loss=testing(resnet50,device,train_loader,n)
                print_testing_accuracy(total_correct,train_loader,batch_size)