from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
import time
import torch.nn.functional as F
import copy

def get_pretrained_model_criterion_optimizer_scheduler():
    model_ft = models.resnet18(pretrained=True)

    # Parameters of newly constructed modules have requires_grad=True by default
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 2)

    criterion = nn.BCEWithLogitsLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.001)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer_ft)

    return model_ft.cuda(), criterion.cuda(), optimizer_ft, exp_lr_scheduler


def train_model(model, criterion, optimizer, scheduler, dataloaders, isRGB = False, num_epochs=7):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0

    dataset_sizes = {'train' : len(dataloaders['train'].dataset), 'val' : len(dataloaders['val'].dataset) }

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            count = 0
            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                #inputs = inputs.to(device)
                #labels = labels.to(device)
                count += 1
                print('All ' + phase + ': ')
                print(len(dataloaders[phase].dataset))
                print('Current ' + phase + ': ')
                print(count)

                # zero the parameter gradients
                optimizer.zero_grad()

                #if not isRGB: inputs = torch.cat([inputs,inputs,inputs], dim = 1)
                inputs = preprocess(inputs, isRGB)

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    print('loss: ' + str(loss.item()))

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        scheduler.step(loss)

                # statistics
                running_loss += loss.item() * inputs.size(0)
                _, comparisonlabel = torch.max(labels,1)
                running_corrects += torch.sum(preds == comparisonlabel)


            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.5f}'.format(
                phase, epoch_loss))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def preprocess(inputs, isRGB):
    if not isRGB: inputs = inputs.repeat(1, 3, 1, 1)
    inputs = inputs - inputs.view(inputs.shape[0], 3, inputs.shape[-1] * inputs.shape[-1]).min(axis=2)[0].reshape(
        inputs.shape[0], 3, 1, 1)
    inputs = inputs / inputs.view(inputs.shape[0], 3, inputs.shape[-1] * inputs.shape[-1]).max(axis=2)[0].reshape(
        inputs.shape[0], 3, 1, 1)
    inputs = F.interpolate(inputs, size=224)
    inputs = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(inputs)
    return inputs