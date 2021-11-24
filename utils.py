'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import os
import sys
import time
import math
import torch.nn as nn
import torch.nn.init as init

    
import torch
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

#transform_crop = transforms.Compose([
#    transforms.RandomCrop(32, padding=4),
#])

class Cifar10Dataset(torchvision.datasets.CIFAR10):
    def __init__(self, root="../data", train=True, download=True, transform=None):
        super().__init__(root=root, train=train, download=download, transform=transform)

    def __getitem__(self, index):
        image, label = self.data[index], self.targets[index]
        
        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]
        return image, label


def getTrainTransforms():
    train_transforms = A.Compose([
         A.RandomCrop(width=32, height=32),
         A.Cutout(num_holes=1, max_h_size=16, max_w_size=16, fill_value=0.47, always_apply=False, p=0.5),
         A.Normalize(mean=(0.49139968, 0.48215827 ,0.44653124), std=(0.24703233,0.24348505,0.26158768)),
         ToTensorV2()
         ])
    return train_transforms

def getTestTransforms():
    test_transforms = A.Compose([
         A.Normalize(mean=(0.49139968, 0.48215827 ,0.44653124), std=(0.24703233,0.24348505,0.26158768)),
         ToTensorV2()
         ])

    return test_transforms



####################### FOR GRAD CAM ########################

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, deprocess_image, preprocess_image

import numpy as np
import cv2

def getTensorImage(img):
    np_image = img.numpy()
    np_image = np.transpose(np_image, (1, 2, 0))
    rgb_img = np.float32(np_image) / 255
    input_tensor = preprocess_image(rgb_img, mean=[0.49139968, 0.48215827 ,0.44653124], 
                                    std=[0.24703233,0.24348505,0.26158768])

    return input_tensor, np_image


#target_layers = [net.layer4[1]]
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import cv2

def getTensorImage(img):
    np_image = img.numpy()
    np_image = np.transpose(np_image, (1, 2, 0))
    rgb_img = np.float32(np_image) / 255
    input_tensor = preprocess_image(rgb_img, mean=[0.49139968, 0.48215827 ,0.44653124], 
                                    std=[0.24703233,0.24348505,0.26158768])

    return input_tensor, np_image



transform_test = transforms.Compose([
    transforms.ToTensor(),
])

# load test data with batch size = 1 for predicting on CPU 
dataloader_args_ = dict(shuffle=False, batch_size=1)
test_ = torchvision.datasets.CIFAR10('./data', train=False, download=True, transform=transform_test)
test_loader_ = torch.utils.data.DataLoader(test_, **dataloader_args_)

test_inf = torchvision.datasets.CIFAR10('./data', train=False, download=True)



def getMissMatchImages(net, target_layers):
    
    mismatch_label_predicted = []
    mismatch_images = []
    mismatch_tgt = []
    mismatch_index = []
    grad_cam_images = []

    device = torch.device("cpu")
    net =  net.to(device)
    net.eval()

    index = 0
    for data_, target_ in test_loader_:
        output_ = net(data_)
        pred = output_.argmax(dim=1, keepdim=True)
    
        if pred.item() != target_.item() :
            image,label  = test_inf[index]
            #mismatch_images.append(image)
            mismatch_label_predicted.append(pred.item())
            mismatch_tgt.append(target_)
            mismatch_index.append(index)
            prob = F.softmax(output_, dim=1)
            pred_prob = prob.data.max(dim=1)[0]
            pred_index = prob.data.max(dim=1)[1]
            #print(pred," ",pred_prob," ",pred_index,"tgt = ",target_)
            cam_image = None
            image_tensor , rgb_img = getTensorImage(torchvision.utils.make_grid(data_))
            target_category = pred
            
            with GradCAM(model=net, target_layers=target_layers, use_cuda=False) as cam:
                cam.batch_size = 128
                grayscale_cam =cam(input_tensor=image_tensor,target_category=target_category,aug_smooth=True,eigen_smooth=True)
                grayscale_cam = grayscale_cam[0, :]
                cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
            
            cam_n_original = np.hstack([image, cam_image])
            mismatch_images.append(image)
            grad_cam_images.append(cam_n_original)

        if len(mismatch_index) == 10:
                break
            
        index += 1 

    return mismatch_tgt, mismatch_label_predicted, mismatch_images, grad_cam_images
    









