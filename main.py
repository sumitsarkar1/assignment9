import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

import sumitEVA7
from sumitEVA7.models import custom_resnet

criterion = nn.CrossEntropyLoss()

def getModel():
    net = custom_resnet.Net()
    return net

def setOptimizer(net, lr):
    lr = 0.012
    optimizer = torch.optim.Adam(net.parameters(), lr)
    return optimizer

def setScheduler(optimizer, epochs, max_lr, steps_per_epoch, pct_start, div_factor, final_div_factor):
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr,steps_per_epoch=steps_per_epoch,
            epochs=epochs, pct_start=pct_start, div_factor=div_factor,final_div_factor=final_div_factor ) 
    return scheduler

from sumitEVA7.utils2 import Cifar10Dataset

class args():
    def __init__(self,device = 'cpu' ,use_cuda = False) -> None:
        self.batch_size = 512
        self.device = device
        self.use_cuda = use_cuda
        self.kwargs = {'num_workers': 1, 'pin_memory': True} if self.use_cuda else {}

        
def getTrainLoader(args):
    train_transforms = sumitEVA7.utils2.getTrainTransforms()
    trainset = Cifar10Dataset(root='../data', train=True,download=True, transform=train_transforms) 
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,shuffle=True, **args.kwargs)
    return trainloader
    
def getTestLoader(args):
    test_transforms = sumitEVA7.utils2.getTestTransforms()
    testset = Cifar10Dataset(root='../data', train=False,download=False, transform=test_transforms) 
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,shuffle=True, **args.kwargs)
    return testloader

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

from tqdm import tqdm

train_losses = []
train_losses_per_epoch = []
test_losses = []
train_acc = []
test_acc = []
train_acc_per_epoch = []
lrs = []


def train(net,optimizer,scheduler, trainloader, device):
    grad_clip = 0.1
    net.train()
    pbar = tqdm(trainloader)
    correct = 0
    processed = 0
    #lrs = []
    
    train_loss_epoch = 0
    correct_epoch = 0
    
    for batch_idx, (data, target) in enumerate(pbar):    
        # get samples
        data, target = data.to(device), target.to(device)
        # Init
        optimizer.zero_grad()
        # Predict
        y_pred = net(data)
        # Calculate loss
        loss = criterion(y_pred, target)
        train_losses.append(loss)
        train_loss_epoch += loss.item()

        # Backpropagation
        loss.backward()

        #nn.utils.clip_grad_value_(net.parameters(), grad_clip)

        optimizer.step()
        
        lrs.append(get_lr(optimizer))
        scheduler.step()

        # Update pbar-tqdm    
        pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)

        pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch_idx} Train Accuracy={100*correct/processed:0.2f}')
        train_acc.append(100*correct/processed)
     
    train_loss_epoch /= len(trainloader.dataset)
    train_losses_per_epoch.append(train_loss_epoch)
    train_acc_per_epoch.append(100. * correct / len(trainloader.dataset))


       
#Testing
def test(net, testloader, device):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    test_loss /= len(testloader.dataset)
    test_losses.append(test_loss)
    
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(testloader.dataset),
        100. * correct / len(testloader.dataset)))
    test_acc.append(100. * correct / len(testloader.dataset))


def getTrainLoss():
    return train_losses_per_epoch

def getTestLoss():
    return test_losses

def getTrainAcc():
    return train_acc_per_epoch

def getTestAcc():
    return test_acc

def getlrVals():
    return lrs




