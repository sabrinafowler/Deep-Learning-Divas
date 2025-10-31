# Deep-Learning-Divas
CSCE 479/879 group work for Sabrina Fowler, Grace Hecke, Abby Veiman, Derek DeBlieck

## Virtual Environment
In order to set up a conda virtual environment using the requirements.txt file, Python version 3.10 should be used.

## Homework 1: Connected Architectures and Fashion MNIST
The assignment for Homework 1 was to design and implement at least 2 architectures for a classification task on the Fashion MNIST dataset. In particular, to load the training data from tensorflow-datasets, partition it into separate training and validation sets, use these to train and evaluate your models, and report the results. 
Model specifications were:
- For each of architecture, use Adam to optimize the parameters 
- For each training run,  use at least two sets of hyperparameters
- Choose at least one regularizer and evaluate system performance with and without it
__Deliverables:__ 
1. A single, well written report (in pdf format) discussing results. The pdf should include the experimental setup, results, and conclusions.
2. Three program files, with the following names:
    - main.py: Code that runs the main loop of training the TensorFlow models
    - model.py: TensorFlow code that defines the network
    - util.py: Helper functions (e.g., for loading the data, small repetitive functions)

## Homework 2: Convolutional Architectures and CIFAR-100
The assignment for Homework 2 was to apply convolutional neural networks to the problem of image classification from the CIFAR-100 dataset. In particular, to designand implement at least two convolutional architectures which satisfy the following specifications:
- Each architecture must use at least two convolutional+pooling layers and at least one connected layer, followed by softmax for the output layer
- Measure loss with cross-entropy
- For each architecture, choose an optimizer
- For each training run use at least two sets of hyperparameters
- Choose a regularizer, in addition to early stopping with a minimum patience hyperparameter value of 5
__Deliverables:__
1. A single, well written report (in pdf format) discussing results. The pdf should include the experimental setup, results, and conclusions.
2. Three program files, with the following names:
    - main.py: Code that runs the main loop of training the TensorFlow models
    - model.py: TensorFlow code that defines the network
    - util.py: Helper functions (e.g., for loading the data, small repetitive functions)