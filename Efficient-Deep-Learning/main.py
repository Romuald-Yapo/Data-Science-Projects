#Import libraries and functions
import matplotlib.pyplot as plt
from torchvision.datasets import CIFAR10
import numpy as np 
import torchvision.transforms as transforms
import torch 
from torch.utils.data.dataloader import DataLoader
from torch import optim, nn
from training_mixup_evaluation import training_mix_up

## Normalization adapted for CIFAR10
normalize_scratch = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

# Transforms is a list of transformations applied on the 'raw' dataset before the data is fed to the network. 
# Here, Data augmentation (RandomCrop and Horizontal Flip) are applied to each batch, differently at each epoch, on the training set data only
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize_scratch,
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    normalize_scratch,
])

### The data from CIFAR10 will be downloaded in the following folder
rootdir = './data/cifar10'

c10train = CIFAR10(rootdir,train=True,download=True,transform=transform_train)
c10test = CIFAR10(rootdir,train=False,download=True,transform=transform_test)

trainloader = DataLoader(c10train,batch_size=32,shuffle=True, num_workers= 4)
testloader = DataLoader(c10test,batch_size=32)  



#Check if we're using a gpu
print(f"Cuda Available : {torch.cuda.is_available()}")


# Create the model 
from resnet import ResNet18

#import training and evaluation function
from training_mixup_evaluation import training_mix_up, evaluation

criterion = nn.CrossEntropyLoss()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''
# Training process

net= ResNet18()
net.to(device)



# create your optimizer and loss function
optimizer = optim.SGD(net.parameters(), lr=0.1)
criterion = nn.CrossEntropyLoss()

n_epochs= 150

#RUN THE TRAINING FUNCTION
train_losses, valid_losses = training_mix_up(n_epochs, trainloader, testloader, net, criterion, optimizer, device)

#plot the obtained losses

plt.plot(range(n_epochs), train_losses)
plt.plot(range(n_epochs), valid_losses)
plt.legend(['train', 'validation'], prop={'size': 10})
plt.title('loss function', size=10)
plt.xlabel('epoch', size=10)
plt.ylabel('loss value', size=10)
plt.savefig('training_valid_loss.png')
plt.show()

'''

'''
#Evaluation process

PATH = "model56_mixup_group_best_model_depruned_quantized.pt"


load_net = ResNet18()

load_net.load_state_dict(torch.load(PATH))

load_net.to(device)

total_params =0

# Count the number of parameters
total_params = sum(p.numel() for p in load_net.parameters())

#compute the non zero params
for p in load_net.parameters():
    total_non_zero_params +=(p.view(-1)!= 0).sum().item()


print("Ratio non zero parameters over number of parameters : ", total_non_zero_params / total_params)

class_names = trainloader.dataset.classes
print(class_names)

#RUN THE EVALUATION FUNCTION
evaluation(load_net, testloader, criterion, device, class_names) 
  

'''
