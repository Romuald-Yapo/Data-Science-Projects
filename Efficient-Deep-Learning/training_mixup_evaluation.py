import numpy as np
import torch
import matplotlib.pyplot as plt
import random
import torch.nn.utils.prune as prune
from torch import nn
import torch.optim.lr_scheduler as lr_schedule
from torch.optim.lr_scheduler import CyclicLR


def training_mix_up(n_epochs, train_loader, valid_loader, model, criterion, optimizer, device): # FUNCTION TO BE COMPLETED
  
  #scheduler = lr_schedule.StepLR(optimizer, step_size=15, gamma=0.1)
  scheduler = CyclicLR(optimizer, base_lr=0.0001, max_lr=0.1,step_size_up=2000, step_size_down=None, mode='triangular')
  

  train_losses, valid_losses = [], []
  valid_loss_min = np.Inf

  for epoch in range(n_epochs):
      train_loss, valid_loss = 0, 0
      
# train the model
      model.train()
      for data, label in train_loader:

          
        #Generate random indices to shuffle the images and target
        indices = torch.randperm(len(data))
        shuffled_data = data[indices]
        shuffled_label = label[indices]

        shuffled_data = shuffled_data.to(device=device, dtype=torch.float32)
        shuffled_label = shuffled_label.to(device=device, dtype=torch.long)
        data = data.to(device=device, dtype=torch.float32)
        label = label.to(device=device, dtype=torch.long)
      
        optimizer.zero_grad() # clear the gradients of all optimized variables
        
        #mix up
        lambda_ = random.random() 
        activation = random.randint(0, 1)
        train_data = MixUp(lambda_, activation, data,shuffled_data)
        
        #training
        output = model(train_data) # forward pass: compute predicted outputs by passing inputs to the model

        loss = lambda_* criterion(output,label) + (1-lambda_) * criterion(output,shuffled_label) # calculate the loss


        loss.backward() # backward pass: compute gradient of the loss with respect to model parameters
        optimizer.step() # perform a single optimization step (parameter update)
        train_loss += loss.item() * data.size(0) # update running training loss
        
      scheduler.step() # scheduling the learning rate

# validate the model
      model.eval()
      for data, label in valid_loader:
          data = data.to(device=device, dtype=torch.float32)
          label = label.to(device=device, dtype=torch.long)
          with torch.no_grad():
              output = model(data)
          loss = criterion(output,label)
          valid_loss += loss.item() * data.size(0)


 # calculate average loss over an epoch
      train_loss /= len(train_loader.sampler)
      valid_loss /= len(valid_loader.sampler)
      train_losses.append(train_loss)
      valid_losses.append(valid_loss)

      print('epoch: {} \ttraining Loss: {:.6f} \tvalidation Loss: {:.6f}'.format(epoch+1, train_loss, valid_loss))

      
# save model if validation loss has decreased
      if valid_loss <= valid_loss_min:
          print('validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
          valid_loss_min,
          valid_loss))
          torch.save(model.state_dict(), 'model'+ str(epoch) +'_mixup_group_4_bf_pruning.pt')
          valid_loss_min = valid_loss
    


  return train_losses, valid_losses

#evaluate the model
def evaluation(model, test_loader, criterion,device, class_names):
  

  test_loss = 0.0
  class_correct = list(0. for i in range(10))
  class_total = list(0. for i in range(10))

  model.eval()
  for data, label in test_loader:
      data = data.to(device=device, dtype=torch.float32)
      label = label.to(device=device, dtype=torch.long)
      with torch.no_grad():
          output = model(data)
      loss = criterion(output, label)
      test_loss += loss.item()*data.size(0)
      _, pred = torch.max(output, 1)
      correct = np.squeeze(pred.eq(label.data.view_as(pred)))
      for i in range(len(label)):
          digit = label.data[i]
          class_correct[digit] += correct[i].item()
          class_total[digit] += 1

  test_loss = test_loss/len(test_loader.sampler)
  print('test Loss: {:.6f}\n'.format(test_loss))
  for i in range(10):
    print('test accuracy of %s: %2d%% (%2d/%2d)' % (class_names[i], 100 * class_correct[i] / class_total[i], np.sum(class_correct[i]), np.sum(class_total[i])))
  print('\ntest accuracy (overall): %2.2f%% (%2d/%2d)' % (100. * np.sum(class_correct) / np.sum(class_total), np.sum(class_correct), np.sum(class_total)))



# mixup
def MixUp (lam, activation, data,shuffled_data) :

    if activation :
        mixedup_data = lam*data + (1 - lam)*shuffled_data

    else :

        mixedup_data = data

    
    return mixedup_data
    
