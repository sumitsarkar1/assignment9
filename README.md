# Create a Custom Resnet Model and train for 90% Test accuracy on Cifar10 Dataset #

## Design Constraints ##
### 1. Model to be used : [custom Resnet18](https://github.com/sumitsarkar1/sumitEVA7/tree/main/models) ###
### 2. Number of epochs to be used: 24 ###
### 3. Learning Rate (LR) Policy : One Cycle LR with max LR at 5th epoch ###
### 4. Data Augmentation to be used : Random Crop, Horizontal Flip, CutOut
### 4. Test Accuracy to achieve : 90% ###

## Final Test Accuracy achieved : 93%+ ##
## Training/Testing Loss and Accuracy , Variation of LR with epochs
![alt text](https://github.com/sumitsarkar1/assignment9/blob/main/plot.jpg)

## Missclassified Images and their gradcam analysis
![alt text](https://github.com/sumitsarkar1/assignment9/blob/main/missclassified_gradcam.jpg)

## Last 4 epoch results ##
```
EPOCH: 21
Loss=0.15072710812091827 Batch_id=97 Train Accuracy=93.65: 100%|█| 98/98 [00:22<
Test set: Average loss: 0.0005, Accuracy: 9315/10000 (93.15%)

EPOCH: 22
Loss=0.12879112362861633 Batch_id=97 Train Accuracy=93.87: 100%|█| 98/98 [00:23<
Test set: Average loss: 0.0005, Accuracy: 9335/10000 (93.35%)

EPOCH: 23
Loss=0.1782904863357544 Batch_id=97 Train Accuracy=94.07: 100%|█| 98/98 [00:23<0
Test set: Average loss: 0.0004, Accuracy: 9327/10000 (93.27%)

EPOCH: 24
Loss=0.1502068042755127 Batch_id=97 Train Accuracy=94.32: 100%|█| 98/98 [00:23<0
Test set: Average loss: 0.0004, Accuracy: 9352/10000 (93.52%)
```
