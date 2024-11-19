#Import libraries and functions
import matplotlib.pyplot as plt
from torchvision.datasets import CIFAR10
import numpy as np 
import torchvision.transforms as transforms
import torch 
from torch.utils.data.dataloader import DataLoader
from torch import nn, optim
from training_mixup import training_mix_up


#Normalization adapted for CIFAR10
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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Check if we're using a gpu
print(f"Cuda Available : {torch.cuda.is_available()}")


# Evaluate our saved model
from models_cifar100.resnet_0 import ResNet18
from training_mixup import training_mix_up, evaluation
import torch.nn.utils.prune as prune


#criterion and class names
class_names = trainloader.dataset.classes
criterion = nn.CrossEntropyLoss()

# Iterative Global Unstructured Pruning
for i in range(5,100,10) :
    #Load model
    PATH = "model145_mixup_group_4_bf_pruning.pt"
    load_net = ResNet18()
    load_net.load_state_dict(torch.load(PATH))
    load_net.to(device)

    #Pruning
    parameters_to_prune =[]
    for m in load_net.modules() :
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            parameters_to_prune.append((m, 'weight'))        
    prune.global_unstructured(parameters= parameters_to_prune, pruning_method= prune.L1Unstructured, amount = i/100)

    #Evaluation
    print('percentage of pruning : ',i)
    evaluation(load_net, testloader, criterion, device, class_names)


'''
#Depruning

for m in load_net.modules():
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
              prune.remove(m, 'weight')   

torch.save(load_net.state_dict(), 'model56_mixup_group_best_model_depruned.pt')


'''

'''
#Pruning 80% and fine tuning

import torch.nn.utils.prune as prune
import torch.nn.functional as F

# Global pruning
PATH = "model145_mixup_group_4_bf_pruning.pt"

load_net = ResNet18()
load_net.load_state_dict(torch.load(PATH))
load_net.to(device)

parameters_to_prune = []
for  m in (load_net.modules()):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        parameters_to_prune.append((m,'weight'))
                    
prune.global_unstructured(parameters= parameters_to_prune, pruning_method= prune.L1Unstructured, amount = 0.80)

torch.save(load_net.state_dict(), 'model145_mixup_group_4_bf_training_80_pruning.pt')


#model training
optimizer = optim.SGD(load_net.parameters(), lr=0.1)
criterion = nn.CrossEntropyLoss()


n_epochs= 250

#RUN THE TRAINING FUNCTION
train_losses, valid_losses = training_mix_up(n_epochs, trainloader, testloader, load_net, criterion, optimizer, device)

#plot the obtained losses


plt.plot(range(n_epochs), train_losses)
plt.plot(range(n_epochs), valid_losses)
plt.legend(['train', 'validation'], prop={'size': 10})
plt.title('loss function', size=10)
plt.xlabel('epoch', size=10)
plt.ylabel('loss value', size=10)
plt.savefig(training_valid_loss_pruning.png)
plt.show()

'''

'''
#16 bits Quantization code

load_net.half()
torch.save(load_net.state_dict(), 'model56_mixup_group_best_model_depruned_quantized.pt')

'''