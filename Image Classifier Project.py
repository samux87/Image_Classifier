
# coding: utf-8

# # Developing an AI application
# 
# Going forward, AI algorithms will be incorporated into more and more everyday applications. For example, you might want to include an image classifier in a smart phone app. To do this, you'd use a deep learning model trained on hundreds of thousands of images as part of the overall application architecture. A large part of software development in the future will be using these types of models as common parts of applications. 
# 
# In this project, you'll train an image classifier to recognize different species of flowers. You can imagine using something like this in a phone app that tells you the name of the flower your camera is looking at. In practice you'd train this classifier, then export it for use in your application. We'll be using [this dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) of 102 flower categories, you can see a few examples below. 
# 
# <img src='assets/Flowers.png' width=500px>
# 
# The project is broken down into multiple steps:
# 
# * Load and preprocess the image dataset
# * Train the image classifier on your dataset
# * Use the trained classifier to predict image content
# 
# We'll lead you through each part which you'll implement in Python.
# 
# When you've completed this project, you'll have an application that can be trained on any set of labeled images. Here your network will be learning about flowers and end up as a command line application. But, what you do with your new skills depends on your imagination and effort in building a dataset. For example, imagine an app where you take a picture of a car, it tells you what the make and model is, then looks up information about it. Go build your own dataset and make something new.
# 
# First up is importing the packages you'll need. It's good practice to keep all the imports at the beginning of your code. As you work through this notebook and find you need to import a package, make sure to add the import up here.

# In[1]:


# Imports here
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")

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


# ## Load the data
# 
# Here you'll use `torchvision` to load the data ([documentation](http://pytorch.org/docs/0.3.0/torchvision/index.html)). The data should be included alongside this notebook, otherwise you can [download it here](https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz). The dataset is split into three parts, training, validation, and testing. For the training, you'll want to apply transformations such as random scaling, cropping, and flipping. This will help the network generalize leading to better performance. You'll also need to make sure the input data is resized to 224x224 pixels as required by the pre-trained networks.
# 
# The validation and testing sets are used to measure the model's performance on data it hasn't seen yet. For this you don't want any scaling or rotation transformations, but you'll need to resize then crop the images to the appropriate size.
# 
# The pre-trained networks you'll use were trained on the ImageNet dataset where each color channel was normalized separately. For all three sets you'll need to normalize the means and standard deviations of the images to what the network expects. For the means, it's `[0.485, 0.456, 0.406]` and for the standard deviations `[0.229, 0.224, 0.225]`, calculated from the ImageNet images.  These values will shift each color channel to be centered at 0 and range from -1 to 1.
#  

# In[2]:


data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'


# In[3]:


# TODO: Define your transforms for the training, validation, and testing sets
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
# TODO: Load the datasets with ImageFolder

# Train
train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
# Validating
valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
# Test
test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

image_datasets = {'train': train_data, 'valid': valid_data, 'test': test_data}

# TODO: Using the image datasets and the trainforms, define the dataloaders
#dataloaders = 

# Tr
trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)

# Va 
validloader = torch.utils.data.DataLoader(valid_data, batch_size=32) # shuffle ?

# Te
testloader = torch.utils.data.DataLoader(test_data, batch_size=32)


# ### Label mapping
# 
# You'll also need to load in a mapping from category label to category name. You can find this in the file `cat_to_name.json`. It's a JSON object which you can read in with the [`json` module](https://docs.python.org/2/library/json.html). This will give you a dictionary mapping the integer encoded categories to the actual names of the flowers.

# In[4]:


import json

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)


# # Building and training the classifier
# 
# Now that the data is ready, it's time to build and train the classifier. As usual, you should use one of the pretrained models from `torchvision.models` to get the image features. Build and train a new feed-forward classifier using those features.
# 
# We're going to leave this part up to you. If you want to talk through it with someone, chat with your fellow students! You can also ask questions on the forums or join the instructors in office hours.
# 
# Refer to [the rubric](https://review.udacity.com/#!/rubrics/1663/view) for guidance on successfully completing this section. Things you'll need to do:
# 
# * Load a [pre-trained network](http://pytorch.org/docs/master/torchvision/models.html) (If you need a starting point, the VGG networks work great and are straightforward to use)
# * Define a new, untrained feed-forward network as a classifier, using ReLU activations and dropout
# * Train the classifier layers using backpropagation using the pre-trained network to get the features
# * Track the loss and accuracy on the validation set to determine the best hyperparameters
# 
# We've left a cell open for you below, but use as many as you need. Our advice is to break the problem up into smaller parts you can run separately. Check that each part is doing what you expect, then move on to the next. You'll likely find that as you work through each part, you'll need to go back and modify your previous code. This is totally normal!
# 
# When training make sure you're updating only the weights of the feed-forward network. You should be able to get the validation accuracy above 70% if you build everything right. Make sure to try different hyperparameters (learning rate, units in the classifier, epochs, etc) to find the best model. Save those hyperparameters to use as default values in the next part of the project.

# In[5]:


# TODO: Build and train your network
# load pre-trained network 
# model = models.densenet121(pretrained=True) 
# model = Network(784, 10, [516, 256], drop_p=0.5)

model = models.vgg16(pretrained=True)

num_nodes_NN = {"vgg16": 25088, "alexnet": 9216, "densenet121":1024, "resnet18": 512}
inputNodes = num_nodes_NN['vgg16']
fc1_nodes = 1000
outputNodes = 102 # 102 flowers categories

# Define a new, untrained feed-forward network as a classifier, using ReLU activations and dropout
# Freeze parameters so we don't backprop through them
for param in model.parameters():
    param.requires_grad = False


classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(inputNodes, fc1_nodes)), 
                          ('relu', nn.ReLU()),
                          ('dropout', nn.Dropout(p=0.5)),
                          ('fc2', nn.Linear(fc1_nodes, outputNodes)),  
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
        
model.classifier = classifier

# print(model)
# model.classifier

# Train the classifier layers using backpropagation and the pre-trained network 
criterion = nn.NLLLoss()
learn_rate = 0.001
n_epochs = 5
print_every = 10
optimizer = optim.Adam(model.classifier.parameters(), lr=learn_rate)

# CPU or GPU switch
cuda = torch.cuda.is_available()
if cuda:
    model.to("cuda")
    device="cuda"
    print('Using GPU')
else:
    model.to("cpu")
    device="cpu"
    print('Using CPU')

# Training

def do_deep_learning(model, trainloader, epochs, print_every, criterion, optimizer, device):
    
    epochs = epochs
    print_every = print_every
    steps = 0

    model.train() 
    
    for e in range(epochs):
        running_loss = 0
        
        
        for ii, (inputs, labels) in enumerate(trainloader):
            steps += 1
            
            # If GPU
            if cuda:
                #inputs, labels = inputs.cuda(), labels.cuda()
                inputs, labels = inputs.to('cuda'), labels.to('cuda')
           
            optimizer.zero_grad()

            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
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
                        if cuda:
                            #inputs_valid, labels_valid = inputs_valid.cuda(), labels_valid.cuda()
                            inputs_valid, labels_valid = inputs_valid.to('cuda:0'), labels_valid.to('cuda:0')
                            model.to('cuda:0')

                        outputs_valid = model.forward(inputs_valid)
                        loss_valid = criterion(outputs_valid, labels_valid)
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
            
    
print("Inizio Allenamento")
do_deep_learning(model, trainloader, n_epochs, print_every, criterion, optimizer, device)
print("Fine Allenamento")


# ## Testing your network
# 
# It's good practice to test your trained network on test data, images the network has never seen either in training or validation. This will give you a good estimate for the model's performance on completely new images. Run the test images through the network and measure the accuracy, the same way you did validation. You should be able to reach around 70% accuracy on the test set if the model has been trained well.

# In[6]:


# TODO: Do validation on the test set 

test_loss = 0
accuracy = 0

model.eval()  

for k, (inputs, labels) in enumerate(testloader):
        
    with torch.no_grad():
        inputs, labels = Variable(inputs), Variable(labels)

        # if GPU active
        if cuda:
            inputs, labels = inputs.cuda(), labels.cuda()
            
        outputs = model.forward(inputs)
        loss = criterion(outputs, labels)
        test_loss += loss.item()

        ps = torch.exp(outputs)
        
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()

print("Test Loss: {:.3f}".format(test_loss))
print("Test Accuracy %: {:.3f}".format(accuracy/len(testloader) * 100))


# ## Save the checkpoint
# 
# Now that your network is trained, save the model so you can load it later for making predictions. You probably want to save other things such as the mapping of classes to indices which you get from one of the image datasets: `image_datasets['train'].class_to_idx`. You can attach this to the model as an attribute which makes inference easier later on.
# 
# ```model.class_to_idx = image_datasets['train'].class_to_idx```
# 
# Remember that you'll want to completely rebuild the model later so you can use it for inference. Make sure to include any information you need in the checkpoint. If you want to load the model and keep training, you'll want to save the number of epochs as well as the optimizer state, `optimizer.state_dict`. You'll likely want to use this trained model in the next part of the project, so best to save it now.

# In[7]:


# TODO: Save the checkpoint 
model.class_to_idx = train_data.class_to_idx

checkpoint = {'class_to_idx': image_datasets['train'].class_to_idx,
              'optimizer': optimizer.state_dict(),
              'classifier': model.classifier,
              'out_HL1': fc1_nodes,
              'learn_rate': learn_rate,
              'state_dict': model.state_dict()}

torch.save(checkpoint, 'checkpoint.pth')
            
print('Modello salvato con le seguenti chiavi:')
print(model.state_dict().keys())


# ## Loading the checkpoint
# 
# At this point it's good to write a function that can load a checkpoint and rebuild the model. That way you can come back to this project and keep working on it without having to retrain the network.

# In[8]:


# TODO: Write a function that loads a checkpoint and rebuilds the model
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = models.vgg16(pretrained=True)
    model.classifier = None # reset classifier
    model.classifier = checkpoint['classifier']
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    
    optimizer.load_state_dict(checkpoint['optimizer'])
    
    return model, optimizer

model, optimizer = load_checkpoint('checkpoint.pth')

print('Modello caricato con le seguenti chiavi:')
print(model.state_dict().keys())


# # Inference for classification
# 
# Now you'll write a function to use a trained network for inference. That is, you'll pass an image into the network and predict the class of the flower in the image. Write a function called `predict` that takes an image and a model, then returns the top $K$ most likely classes along with the probabilities. It should look like 
# 
# ```python
# probs, classes = predict(image_path, model)
# print(probs)
# print(classes)
# > [ 0.01558163  0.01541934  0.01452626  0.01443549  0.01407339]
# > ['70', '3', '45', '62', '55']
# ```
# 
# First you'll need to handle processing the input image such that it can be used in your network. 
# 
# ## Image Preprocessing
# 
# You'll want to use `PIL` to load the image ([documentation](https://pillow.readthedocs.io/en/latest/reference/Image.html)). It's best to write a function that preprocesses the image so it can be used as input for the model. This function should process the images in the same manner used for training. 
# 
# First, resize the images where the shortest side is 256 pixels, keeping the aspect ratio. This can be done with the [`thumbnail`](http://pillow.readthedocs.io/en/3.1.x/reference/Image.html#PIL.Image.Image.thumbnail) or [`resize`](http://pillow.readthedocs.io/en/3.1.x/reference/Image.html#PIL.Image.Image.thumbnail) methods. Then you'll need to crop out the center 224x224 portion of the image.
# 
# Color channels of images are typically encoded as integers 0-255, but the model expected floats 0-1. You'll need to convert the values. It's easiest with a Numpy array, which you can get from a PIL image like so `np_image = np.array(pil_image)`.
# 
# As before, the network expects the images to be normalized in a specific way. For the means, it's `[0.485, 0.456, 0.406]` and for the standard deviations `[0.229, 0.224, 0.225]`. You'll want to subtract the means from each color channel, then divide by the standard deviation. 
# 
# And finally, PyTorch expects the color channel to be the first dimension but it's the third dimension in the PIL image and Numpy array. You can reorder dimensions using [`ndarray.transpose`](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.ndarray.transpose.html). The color channel needs to be first and retain the order of the other two dimensions.

# In[9]:


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # TODO: Process a PIL image for use in a PyTorch model
    
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


# To check your work, the function below converts a PyTorch tensor and displays it in the notebook. If your `process_image` function works, running the output through this function should return the original image (except for the cropped out portions).

# In[10]:


def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

image_path = 'flowers/test/1/image_06764.jpg'   
image = Image.open(image_path)

imshow(process_image(image))


# ## Class Prediction
# 
# Once you can get images in the correct format, it's time to write a function for making predictions with your model. A common practice is to predict the top 5 or so (usually called top-$K$) most probable classes. You'll want to calculate the class probabilities then find the $K$ largest values.
# 
# To get the top $K$ largest values in a tensor use [`x.topk(k)`](http://pytorch.org/docs/master/torch.html#torch.topk). This method returns both the highest `k` probabilities and the indices of those probabilities corresponding to the classes. You need to convert from these indices to the actual class labels using `class_to_idx` which hopefully you added to the model or from an `ImageFolder` you used to load the data ([see here](#Save-the-checkpoint)). Make sure to invert the dictionary so you get a mapping from index to class as well.
# 
# Again, this method should take a path to an image and a model checkpoint, then return the probabilities and classes.
# 
# ```python
# probs, classes = predict(image_path, model)
# print(probs)
# print(classes)
# > [ 0.01558163  0.01541934  0.01452626  0.01443549  0.01407339]
# > ['70', '3', '45', '62', '55']
# ```

# In[11]:


def predict(image_path, model, topk=5):
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

# Test the prediction function

image_path = 'flowers/test/1/image_06743.jpg' 
probs, classes = predict(image_path, model)

print(probs)
print(classes)

labels = []
for i in range(len(classes)):
    labels.append(cat_to_name[classes[i]])
    
print(labels)
    


# ## Sanity Checking
# 
# Now that you can use a trained model for predictions, check to make sure it makes sense. Even if the testing accuracy is high, it's always good to check that there aren't obvious bugs. Use `matplotlib` to plot the probabilities for the top 5 classes as a bar graph, along with the input image. It should look like this:
# 
# <img src='assets/inference_example.png' width=300px>
# 
# You can convert from the class integer encoding to actual flower names with the `cat_to_name.json` file (should have been loaded earlier in the notebook). To show a PyTorch tensor as an image, use the `imshow` function defined above.

# In[14]:


# TODO: Display an image along with the top 5 classes

def sanityCheck(image_path):
    # Run the prediction engine
    probs, classes = predict(image_path, model)

    # Get the labels
    labels = []
    for i in range(len(classes)):
        labels.append(cat_to_name[classes[i]])

    print('File path ' + image_path)
    print('\nMost likely classification: ' + (labels[0]).title() + ' (' + str(round(probs[0]*100)) + '%)')

    # Display the chart
    currentflower = pd.DataFrame({'Probability': probs, 'Flower Classification': labels})
    currentflower.set_index('Flower Classification')
    sb.barplot(data=currentflower, y = 'Flower Classification', x= 'Probability', color= sb.color_palette()[0])
    
    # Display the image
    image = Image.open(image_path)
    processed_image = process_image(image)
    imshow(processed_image)
    
# Get a random image from the Test folder
randomfolder = random.choice(os.listdir('flowers/test'))
randomfile = random.choice(os.listdir('flowers/test/' + str(randomfolder)))

randomfilepath = 'flowers/test/' + str(randomfolder) + '/' + str(randomfile)

# Run the sanity check
sanityCheck(randomfilepath)

