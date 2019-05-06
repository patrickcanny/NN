# NN
EECS 738 - Implementing Neural Networks from Scratch

Patrick Canny & Liam Ormiston

## To Run:
1. Clone this repository locally, and navigate into it.

2. `cd src/`

3. `make`

## Background
The ultimate test of our machine learning prowess. Neural Networks are really hot right now, so making one from scratch seems like a good idea. We hope to learn which Neural Net is the easiest to understand and explain to someone else.

## Ideas
We started by looking at a simple NN from a medium article that we found. Check out the NN.py file for this implementation in order to get our bearings. The article is here: [How to build your own Neural Network from scratch in Python](https://towardsdatascience.com/how-to-build-your-own-neural-network-from-scratch-in-python-68998a08e4f6). We ended up implementing a single-layer perceptron. 

## Datasets
The Tox21 data set comprises 12,060 training samples and 647 test samples that represent chemical compounds. There are 801 "dense features" that represent chemical descriptors, such as molecular weight, solubility or surface area, and 272,776 "sparse features" that represent chemical substructures (ECFP10, DFS6, DFS8; stored in Matrix Market Format ). The purpose of this dataset is to predict the outcome of biological assays.

## Conclusions/Findings
Our neural network was fairly accurate despite having few layers. The name of the neural net that we made was Perceptron. With the current number of epochs, we get an ~88% accuracy score. We could potentially increase accuracy by adding more epochs or changing the architecture of the neural network (more layers or different activation function). 
