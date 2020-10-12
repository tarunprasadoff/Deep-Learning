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
write = 'Results/2'

models = 'Models/2/model-'
ext = '.pth'

trainName = write + '/train.csv'
classTrainName = write + '/classTrain.csv'

pd.DataFrame(columns = ['epoch', 'accuracy']).to_csv(trainName, index=False)
pd.DataFrame(columns = ['epoch', 'aeroplane', 'cat', 'deer', 'dog', 'frog']).to_csv(classTrainName, index=False)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=4, stride=2)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2)
        self.fc1 = nn.Linear(in_features=256, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=32)
        self.fc3 = nn.Linear(in_features=32, out_features=5)      # change out_features according to number of classes

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.avg_pool2d(x, kernel_size=x.shape[2:])
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

num_epochs = 15

########################################################################
# The output of torchvision datasets are PILImage images of range [0, 1].

# Apply necessary image transfromations here 

transform = transforms.Compose([ #torchvision.transforms.RandomHorizontalFlip(p=0.5),
                                #torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(0.1, 0.1), scale=(0.8, 1.2)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5, 0.5, 0.5])])
print(transform)

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

def eval(loader, evalNet):
  correct = 0
  total = 0
  with torch.no_grad():
      for data in tqdm(loader):
          images, labels = data
          if torch.cuda.is_available():
              images, labels = images.cuda(), labels.cuda()        
          outputs = evalNet(images)
          _, predicted = torch.max(outputs.data, 1)
          total += labels.size(0)
          correct += (predicted == labels).sum().item()
  return (100 * correct / total)

def find_classes():
    classes = ['aeroplane', 'cat', 'deer', 'dog', 'frog']
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

def classwise_test(loader, net):
########################################################################
# class-wise accuracy

    classes, _ = find_classes()
    n_class = len(classes) # number of classes

    class_correct = list(0. for i in range(n_class))
    class_total = list(0. for i in range(n_class))
    with torch.no_grad():
        for data in tqdm(loader):
            images, labels = data
            if torch.cuda.is_available():
                images, labels = images.cuda(), labels.cuda()        
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i].item()
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(n_class):
        print('Accuracy of %10s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))
        
    return (100 * class_correct[0] / class_total[0]), (100 * class_correct[1] / class_total[1]), (100 * class_correct[2] / class_total[2]), (100 * class_correct[3] / class_total[3]), (100 * class_correct[4] / class_total[4])

def test(isTest, epoch):

    net = Net()

    # transfer the model to GPU
    if torch.cuda.is_available():
        net = net.cuda()

    net.load_state_dict(torch.load( models + str(epoch) + ext ))

    perc = eval(trainloader, net)
    with open(trainName, 'a') as newFile:
        newFileWriter = csv.writer(newFile)
        newFileWriter.writerow([epoch+1, perc])
        print('Accuracy of the network on the train images: ' + str(perc))
    with open(classTrainName, 'a') as newFile:
        class0, class1, class2, class3, class4 = classwise_test(trainloader, net)
        newFileWriter = csv.writer(newFile)
        newFileWriter.writerow([epoch+1, class0, class1, class2, class3, class4])

for epoch in range(num_epochs):  # loop over the dataset multiple times
    print('epoch ', epoch + 1)
    test(False, epoch) 

print('Finished')