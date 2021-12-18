# Create a Custom Resnet Model and train for 90% Test accuracy #

## Design Constraints ##
### 1. Model to be used : [custom Resnet18](https://github.com/sumitsarkar1/sumitEVA7/tree/main/models) ###
### 2. Number of epochs to be used: 24 ###
### 3. Learning Rate (LR) Policy : One Cycle LR with max LR at 5th epoch ###
### 4. Data Augmentation to be used : Random Crop, Horizontal Flip, CutOut
### 4. Test Accuracy to achieve : 90% ###

## Final Test Accuracy achieved : 93%+ ##
## Training/Testing Loss and Accuracy , Variation of LR with eochs
![alt text](https://github.com/sumitsarkar1/assignment9/blob/main/plot.jpg)

## Missclassified Images and their gradcam analysis
![alt text](https://github.com/sumitsarkar1/assignment9/blob/main/missclassified_gradcam.jpg)

