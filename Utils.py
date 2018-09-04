# Created by Samuele Buosi 25/8/18
import matplotlib.pyplot as plt

import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

from collections import OrderedDict
from torch.autograd import Variable

from PIL import Image
import os, random
import pandas as pd
import seaborn as sb

#Load the dataSet
def load_dataSet(path  = "./flowers"):
    '''
    Arguments : dataSet path
    Returns : The datasets loaders for train, validation and test + image_datasets
    This function creates the 3 loaders necessery for the NN
    '''
    data_dir = path
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # Define transforms for the training, validation, and testing sets
    # Training
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224), # 224x224 images 
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

    # Validating 
    valid_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])
    # Test
    test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])
    # Loading the datasets with ImageFolder

    # Train
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    # Validating
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    # Test
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

    image_datasets = {'train': train_data, 'valid': valid_data, 'test': test_data}

    # Using the image datasets and the trainforms, define the dataloaders 
    # Tr
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    # Va 
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=32) # shuffle ?
    # Te
    testloader = torch.utils.data.DataLoader(test_data, batch_size=32)

    return trainloader , validloader, testloader, image_datasets

# init the NN
def nn_init(arch='vgg16', dropout=0.5, fc1_nodes = 1000, outputNodes = 102, learn_rate = 0.001, gpu='cpu' ):
    '''
    Arguments : The architecture for the network (alexnet,densenet121,vgg16, resnet18) and 
                the hyperparameters for the network (dropout, hidden layer 1 nodes, output nodes , learning rate) 
                + gpu flag
    Returns : A configurated model + criterion and optimizer
    '''
    
    # load pre-trained network 
    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif arch == 'densenet121':
        model = models.densenet121(pretrained=True)
    elif arch == 'alexnet':
        model = models.alexnet(pretrained = True)
    elif arch == 'resnet18':
        model = models.resnet18(pretrained = True)
    else:
        print("{} is not a valid model. NN accepted: vgg16,densenet121 ,alexnet or resnet18.".format(arch))
        # break ?

    num_nodes_NN = {"vgg16": 25088, "alexnet": 9216, "densenet121":1024, "resnet18": 512}
    inputNodes = num_nodes_NN[arch]

    # Define a new, untrained feed-forward network as a classifier, using ReLU activations and dropout
    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(inputNodes, fc1_nodes)), 
                          ('relu', nn.ReLU()),
                          ('dropout', nn.Dropout(dropout)),
                          ('fc2', nn.Linear(fc1_nodes, outputNodes)),  
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
        
    model.classifier = classifier
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learn_rate)
    #model.class_to_idx = train_data.class_to_idx
            
    # GPU switch 
    if torch.cuda.is_available() and gpu == 'gpu':
        model.to("cuda")
        print('Using GPU')
    else: 
        model.to("cpu")
        print('Using CPU')
        
    return model, criterion, optimizer 

# NN Training
def do_deep_learning(model, trainloader, validloader, epochs, print_every, criterion, optimizer, gpu):
    
    steps = 0
    cuda = torch.cuda.is_available()
    
    print('NN Training Start')    
    model.train() 
   
    for e in range(epochs):
        running_loss = 0
        
        for ii, (inputs, labels) in enumerate(trainloader):
            steps += 1
            
            # If GPU
            if cuda and gpu == 'gpu':
                #inputs, labels = inputs.cuda(), labels.cuda()
                inputs, labels = inputs.to('cuda'), labels.to('cuda')
            
            optimizer.zero_grad()

            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion (outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            if steps % print_every == 0:
                
                # Validazione
                model.eval()
            
                with torch.no_grad():
                    valid_loss = 0
                    accuracy = 0
                    for j, (inputs_valid, labels_valid) in enumerate(validloader):

                        inputs_valid, labels_valid = Variable(inputs_valid), Variable(labels_valid)

                        # if GPU
                        if cuda and gpu == 'gpu':
                            #inputs_valid, labels_valid = inputs_valid.cuda(), labels_valid.cuda()
                            inputs_valid, labels_valid = inputs_valid.to('cuda:0'), labels_valid.to('cuda:0')
                            model.to('cuda:0')

                        outputs_valid = model.forward(inputs_valid)
                        loss_valid = criterion (outputs_valid, labels_valid)
                        valid_loss += loss_valid.item()

                        ps = torch.exp(outputs_valid)
                        equality = (labels_valid.data == ps.max(dim=1)[1])
                        accuracy += equality.type(torch.FloatTensor).mean()
                
                print("Epoch: {}/{} --- ".format(e+1, epochs),
                      "Training Loss: {:.4f} --- ".format(running_loss/print_every),
                      "Valid Loss: {:.4f} --- ".format(valid_loss/len(validloader)),
                      "Valid Accuracy %: {:.4f}".format(accuracy/len(validloader) * 100))
  
                running_loss = 0
                
                # return to training mode
                model.train()
                
    print('NN Training Finish')
    
# Save NN state (checkpoint)  
def save_checkpoint(path, model, optimizer, epochs, fc1_nodes, learn_rate, arch):
    '''
    Arguments: The saving path, model, epochs, fc1_nodes, arch
    Returns: Nothing
    This function saves the model at a specified path by the user
    '''
    # TODO: check train_data variable
    # model.class_to_idx = train_data.class_to_idx
    # 'class_to_idx': image_datasets['train'].class_to_idx
    
    # Reset to CPU
    model.to("cpu")
    
    checkpoint = {
              'epochs': epochs,
              'optimizer': optimizer.state_dict(),
              'classifier': model.classifier,
              'out_HL1': fc1_nodes,
              'learn_rate': learn_rate,
              'state_dict': model.state_dict(),
              'class_to_idx':model.class_to_idx,
              'arch': arch}
   
    torch.save(checkpoint, 'checkpointpy.pth')
              
# Load NN checkpoint
def load_checkpoint(filepath, gpu):
    checkpoint = torch.load(filepath)
    
    # Load Arch
    arch = checkpoint['arch']
    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif arch == 'densenet121':
        model = models.densenet121(pretrained=True)
    elif arch == 'alexnet':
        model = models.alexnet(pretrained = True)
    elif arch == 'resnet18':
        model = models.resnet18(pretrained = True)
    else:
        print("{} is not a valid model. NN accepted: vgg16,densenet121 ,alexnet or resnet18.".format(arch))
        
    model.classifier = None # reset classifier
    model.classifier = checkpoint['classifier']
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    
    optimizer = optim.Adam(model.classifier.parameters(), lr=checkpoint['learn_rate'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    
    epochs = checkpoint['epochs']
    
    # GPU switch 
    if torch.cuda.is_available() and gpu == 'gpu':
        model.to("cuda")
        print('Using GPU')
    else: 
        model.to("cpu")
        print('Using CPU')
        
    return model, optimizer, epochs

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # Process a PIL image for use in a PyTorch model
    
    #Resize the images where the shortest side is 256 pixels, keeping the aspect ratio.
    factor =  256 / min(image.width, image.width)
    
    if image.width < image.height:
        new_width = 256
        new_height = round(image.height * factor)
    else: #height < width
        new_height = 256
        new_width = round(image.width * factor)
    
    image = image.resize((new_height, new_width))

    #Define the center box and crop
    boxlen = (256 - 224) / 2
    box = boxlen, boxlen, 256 - boxlen, 256 - boxlen
    image = image.crop(box)
    
    #Colour adjustment
    np_image = np.array(image)
    np_image = np_image / 255
    
    #Do image normalization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std
    
    np_image = np_image.transpose((2, 0, 1))
    
    return np_image

def predict(image_path, model, topk):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # TODO: Implement the code to predict the class from an image file
    
    cuda = torch.cuda.is_available()
    if cuda:
        model.to("cuda:0")
        device="cuda"
    else:
        model.to("cpu")
        device="cpu"

    model.eval()  

    image = Image.open(image_path)
    processed_image = process_image(image)
    processed_image = torch.from_numpy(np.array([processed_image])).float() # Convert to Tensor
    
    inputs = Variable(processed_image)
    
    # if GPU
    if cuda:
        inputs = inputs.cuda()

    outputs = model.forward(inputs)
    ps = torch.exp(outputs)
    
    torchreturn = torch.topk(ps, topk)
    probs = torchreturn[0]
    indexes = torchreturn[1]
    
    # Extract the values we need
    probs = probs.tolist()[0]
    indexes = indexes.tolist()[0]
     
    conersionindexes = []
    for i in range(len(model.class_to_idx.items())):
        conersionindexes.append(list(model.class_to_idx.items())[i][0])

    classes = []
    for i in range(topk):
        classes.append(conersionindexes[indexes[i]])
        
    return probs, classes



            





    
    
    

