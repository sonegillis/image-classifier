from arg_parser import get_train_args
from utils import Toolkits

import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

from collections import OrderedDict
import sys
import json

drop_p = 0.5
input_size = 25088
output_size = 102
learning_rate, hidden_units, epochs, gpu, arch, save_dir, data_dir = Toolkits.initialiseTrainArgs(get_train_args())
    
def getLoaders(train_dir, valid_dir, test_dir):
    train_transforms = transforms.Compose([
                                            transforms.Resize(256),
                                            transforms.CenterCrop(224),
                                            transforms.RandomRotation(30),
                                            transforms.RandomResizedCrop(224),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    valid_transforms = transforms.Compose([
                                            transforms.Resize(256),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_datasets = datasets.ImageFolder(train_dir, train_transforms)
    valid_datasets = datasets.ImageFolder(valid_dir, valid_transforms)
    test_datasets = datasets.ImageFolder(test_dir, valid_transforms)

    train_loaders = torch.utils.data.DataLoader(train_datasets, batch_size=16, shuffle=True)
    valid_loaders = torch.utils.data.DataLoader(valid_datasets, batch_size=16, shuffle=True)
    test_loaders = torch.utils.data.DataLoader(test_datasets, batch_size=16, shuffle=True)
    
    return train_loaders, valid_loaders, test_loaders

def getCatNames():
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    
    return cat_to_name

def validation(model, valid_loader, criterion):
    accuracy = 0
    test_loss = 0
    for images, labels in valid_loader:
        images, labels = images.to('cuda'), labels.to('cuda')
        outputs = model.forward(images)
        test_loss += criterion(outputs, labels)
        
        ps = torch.exp(outputs)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
        
    return test_loss, accuracy

def build_classifier(input_size, hidden_layers, output_size, drop_p=0.3):
    ordered_dict = OrderedDict()
    layers = [input_size] + hidden_layers
    layer_sizes = zip(layers[:-1], layers[1:])
    print(layer_sizes)
    for ii, layer in enumerate(layer_sizes):
        print(layer)
        ordered_dict["fc" + str(ii+1)] = nn.Linear(layer[0], layer[1])
        ordered_dict["relu" + str(ii+1)] = nn.ReLU()
        ordered_dict["dropout" + str(ii+1)] = nn.Dropout(drop_p)
        
    ordered_dict["fc" + str(len(layers) + 1)] = nn.Linear(layers[-1], output_size)
    ordered_dict["output"] = nn.LogSoftmax(dim=1)
    print(ordered_dict)
    classifier = nn.Sequential(ordered_dict)
    return classifier
      
def get_pretrained_model(arch):
    arch_dict = {
        "vgg13": models.vgg13(pretrained=True),
        "vgg16": models.vgg16(pretrained=True)
    }
    model = arch_dict[arch]
    
    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False
    return model

def save_checkpoint(input_size, hidden_layers, output_size, optimizer, save_dir):
    model.class_to_idx = train_datasets.class_to_idx
    checkpoint = {'input_size': input_size,
              'output_size': output_size,
              'hidden_layers': hidden_layers,
              'class_to_idx': train_datasets.class_to_idx,
              'model_state_dict': model.state_dict(),
              'optimizer_state_dict': optimizer.state_dict()}
    
    torch.save(checkpoint, data_dir + 'checkpoint.pth')
    
def train(model, epochs, learning_rate, train_loaders, valid_loaders, criterion, device):
    model.to(device)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    count = 0
    for e in range(epochs):
        train_loss = 0
        print("Start Epoch #{}".format(e+1))
        for ii, (inputs, labels) in enumerate(train_loaders):
            model.train()
            count += 1
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            logits = model.forward(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            if (count % 20 == 0):
                # Make sure network is in eval mode for inference
                model.eval()

                # Turn off gradients for validation, saves memory and computations
                with torch.no_grad():
                    test_loss, accuracy = validation(model, valid_loaders, criterion)

                print("Epoch: {}/{}.. ".format(e+1, epochs),
                      "Training Loss: {:.3f}.. ".format(train_loss/count),
                      "Test Loss: {:.3f}.. ".format(test_loss/len(valid_loaders)),
                      "Test Accuracy: {:.3f}".format(accuracy/len(valid_loaders)))

                train_loss = 0
    if save_dir:
        save_checkpoint(input_size, hidden_layers, output_size, optimizer, save_dir)
    
def main():
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    train_loaders, valid_loaders, test_loader = getLoaders(train_dir, valid_dir, test_dir)
    
    classifier = build_classifier(input_size, hidden_units, output_size)
    
    model = get_pretrained_model(arch)
    model.classifier = classifier
    
    device = "cuda"
    if not gpu:
        device = "cpu"
        
    train(model, epochs, learning_rate, train_loaders, valid_loaders, nn.NLLLoss(), device)
    
if __name__ != 'main':
    main()
