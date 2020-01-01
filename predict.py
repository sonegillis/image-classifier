from arg_parser import get_predict_args
from utils import Toolkits

import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

from collections import OrderedDict
import json

inputs, checkpoint, top_k, gpu, category_names = Toolkits.initialisePredictArgs(get_predict_args())

with open(category_names, 'r') as f:
    cat_to_name = json.load(f)

device = "cpu"
if gpu:
    device = "cuda"
    
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
    
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    hdl = checkpoint['hidden_layers']
    input_size = checkpoint['input_size']
    output_size = checkpoint['output_size']
    class_to_idx = checkpoint['class_to_idx']
    
    model = models.vgg16(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    
    classifier = build_classifier(input_size, hdl, output_size)
    
    model.classifier = classifier
    model.class_to_idx = class_to_idx
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model = model.to(device)
    
    return model

def predict(image_path, model, top_k):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    with torch.no_grad():
        image = Toolkit.process_image(image_path)
        print(image.shape)
        image.unsqueeze_(0)
        output = model.forward(image)
        ps = torch.exp(output)
        result = ps.topk(5)
        return result

def main():
    model = load_checkpoint(checkpoint).to(device)
    results = predict(inputs, model, top_k)
    print("results is ", results)
    idx_to_class = {model.class_to_idx[key]:key for key in model.class_to_idx.keys()}
    
    probs, classes = result[0].view(-1).numpy(), result[1].view(-1).numpy()
    classes_name = [cat_to_name[idx_to_class[idx]] for idx in classes]
    
if __name__ != 'main':
    main()