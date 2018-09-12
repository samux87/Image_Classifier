# AI Programming with Python Project - Image Classifier

This is a image classifier coded for the "Udacity's AI Programming with Python Nanodegree" program using neural networks.

The Image Classifier recognize different species of flowers (102 classes).

In this project, i first develop code for an image classifier built with PyTorch, then i converted it into a command line application.

## Jupyter Notebook

For a first look of the Jupyter Notebook, i suggest to use [nbviewer](https://nbviewer.jupyter.org/github/samux87/Image_Classifier/blob/master/Image%20Classifier%20Project.ipynb)

Otherwise, you can open it in [pdf](https://github.com/samux87/Image_Classifier/blob/master/Image%20Classifier%20Project.pdf) format.

## Python script usage  
There are three files:
* train.py (init a new NN, train it and make a checkpoint)
    * Basic Usage : ```python train.py data_directory --gpu gpu```
    * arguments:
        * ```---arch``` (vgg16, vgg13, alexnet, resnet18, densenet121)
        * ```---hidden_units1```
        * ```---learning_rate```
        * ```---epochs```
        * ```---gpu```
        * ```---dropout```
* predict.py (load a checkpoint and Predict)
    * Basic usage: ```python predict.py /path/to/image checkpoint --gpu gpu```
    * arguments:
        * ```---topk```
        * ```---category_names```
        * ```---gpu```

* utils.py (support library)

## Authors

* **Samuele Buosi**
* **Udacity**

This project is licensed under the MIT License - see the [LICENSE.md](https://github.com/samux87/Image_Classifier/blob/master/LICENSE.md) file for details
