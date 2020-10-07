import torch, os
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

import pandas as pd
import csv

read = '~/Data'
modelWrite = 'Models/2'
resultWrite = 'Results/2'

########################################################################
# The output of torchvision datasets are PILImage images of range [0, 1].

# Apply necessary image transfromations here 

transform = transforms.Compose([ #torchvision.transforms.RandomHorizontalFlip(p=0.5),
                                #torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(0.1, 0.1), scale=(0.8, 1.2)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5, 0.5, 0.5])])

batch_size =  4

train_data_dir = read + '/train' # put path of training dataset
val_data_dir = read + '/val' # put path of validation dataset
test_data_dir = read + '/test' # put path of test dataset

trainset = torchvision.datasets.ImageFolder(root= train_data_dir, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

valset = torchvision.datasets.ImageFolder(root= val_data_dir, transform=transform)
valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

testset = torchvision.datasets.ImageFolder(root= test_data_dir, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

trainName = resultWrite + '/train.csv'
testName = resultWrite + '/test.csv'
classTestName = resultWrite + '/classTest.csv'
valName = resultWrite + '/val.csv'
classValName = resultWrite + '/classVal.csv'

pd.DataFrame(columns = ['epoch', 'training_loss']).to_csv(trainName, index=False)
pd.DataFrame(columns = ['epoch', 'accuracy']).to_csv(valName, index=False)
pd.DataFrame(columns = ['accuracy']).to_csv(testName, index=False)
pd.DataFrame(columns = ['epoch', 'aeroplane', 'cat', 'deer', 'dog', 'frog']).to_csv(classValName, index=False)
pd.DataFrame(columns = ['aeroplane', 'cat', 'deer', 'dog', 'frog']).to_csv(classTestName, index=False)

########################################################################
# Define a Convolution Neural Network
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Copy the neural network from the Neural Networks section before and modify it to
# take 3-channel images (instead of 1-channel images as it was defined).



# <<<<<<<<<<<<<<<<<<<<< EDIT THE MODEL DEFINITION >>>>>>>>>>>>>>>>>>>>>>>>>>
# Try experimenting by changing the following:
# 1. number of feature maps in conv layer
# 2. Number of conv layers
# 3. Kernel size
# etc etc.,

num_epochs = 15         # desired number of training epochs.
learning_rate = 0.01   

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=4, stride=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2)
        self.fc1 = nn.Linear(in_features=256, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=32)
        self.fc3 = nn.Linear(in_features=32, out_features=5)      # change out_features according to number of classes

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = F.avg_pool2d(x, kernel_size=x.shape[2:])
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

################### DO NOT EDIT THE BELOW CODE!!! #######################

#net = ResNet()
net = Net()

# transfer the model to GPU
if torch.cuda.is_available():
    net = net.cuda()

########################################################################
# Define a Loss function and optimizer
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Let's use a Classification Cross-Entropy loss and SGD with momentum.

import torch.optim as optim
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)

num_params = np.sum([p.nelement() for p in net.parameters()])
print(num_params, ' parameters')

########################################################################
# Train the network
# ^^^^^^^^^^^^^^^^^^^^

def train(epoch, trainloader, optimizer, criterion):
    running_loss = 0.0
    for i, data in enumerate(tqdm(trainloader), 0):
        # get the inputs
        inputs, labels = data
        if torch.cuda.is_available():
            inputs, labels = inputs.cuda(), labels.cuda()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
    with open(trainName, 'a') as newFile:
      newFileWriter = csv.writer(newFile)
      newFileWriter.writerow([epoch+1, running_loss / (len(trainloader))])
      print('epoch %d training loss: %.3f' %(epoch + 1, running_loss / (len(trainloader))))
    
########################################################################
# Let us look at how the network performs on the test dataset.

def test(testloader, model, isVal, epoch):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in tqdm(testloader):
            images, labels = data
            if torch.cuda.is_available():
                images, labels = images.cuda(), labels.cuda()        
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    if isVal:
      with open(valName, 'a') as newFile:
        newFileWriter = csv.writer(newFile)
        newFileWriter.writerow([epoch+1, (100 * correct / total)])
    else:
      with open(testName, 'a') as newFile:
        newFileWriter = csv.writer(newFile)
        newFileWriter.writerow([(100 * correct / total)])
        
    print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))

#########################################################################
# get details of classes and class to index mapping in a directory
def find_classes(dir):
    classes = ['aeroplane', 'cat', 'deer', 'dog', 'frog']
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def classwise_test(testloader, model, isVal, epoch):
########################################################################
# class-wise accuracy

    classes, _ = find_classes(train_data_dir)
    n_class = len(classes) # number of classes

    class_correct = list(0. for i in range(n_class))
    class_total = list(0. for i in range(n_class))
    with torch.no_grad():
        for data in tqdm(testloader):
            images, labels = data
            if torch.cuda.is_available():
                images, labels = images.cuda(), labels.cuda()        
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(labels.size(0)):
                label = labels[i].item()
                class_correct[label] += c[i].item()
                class_total[label] += 1

    if isVal:
      with open(classValName, 'a') as newFile:
        newFileWriter = csv.writer(newFile)
        newFileWriter.writerow([epoch+1, 100 * class_correct[0] / class_total[0], 100 * class_correct[1] / class_total[1], 100 * class_correct[2] / class_total[2], 100 * class_correct[3] / class_total[3], 100 * class_correct[4] / class_total[4]])
    else:
      with open(classTestName, 'a') as newFile:
        newFileWriter = csv.writer(newFile)
        newFileWriter.writerow([100 * class_correct[0] / class_total[0], 100 * class_correct[1] / class_total[1], 100 * class_correct[2] / class_total[2], 100 * class_correct[3] / class_total[3], 100 * class_correct[4] / class_total[4]])    

    for i in range(n_class):
        print('Accuracy of %10s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))

print('Start Training')

for epoch in range(num_epochs):  # loop over the dataset multiple times
    print('epoch ', epoch + 1)
    train(epoch, trainloader, optimizer, criterion)
    test(valloader, net, True, epoch)
    classwise_test(valloader, net, True, epoch)
    # save model checkpoint 
    torch.save(net.state_dict(), modelWrite+'/model-'+str(epoch)+'.pth')      

print('performing test')
test(testloader, net, False, None)
classwise_test(testloader, net, False, None)

print('Finished Training')