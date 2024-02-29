# Brain Tumor Detection Project

This project aims to detect brain tumors from MRI images using Convolutional Neural Networks (CNN). Two different CNN models were implemented: a simple CNN model and a VGG16-inspired CNN model.

## Acknowledgement
We would like to acknowledge the contributions of [insert names or organizations here] for their support and assistance in developing this project.

## Badges
[![Python Version](https://img.shields.io/badge/python-3.8.18-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow Version](https://img.shields.io/badge/tensorflow-2.9.1-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![GitHub Repo](https://img.shields.io/badge/github-repo-blueviolet.svg)](https://github.com/yourusername/your-repo)
[![Scikit-learn Version](https://img.shields.io/badge/scikit--learn-1.4-yellowgreen.svg)](https://scikit-learn.org/)

## Dataset
The dataset used for this project can be found [here](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset/data).



## Key Terms
### Imaging Techniques
- MRI: Magnetic Resonance Imaging - A medical imaging technique used to visualize internal structures of the body in detail.

### Neural Network Architectures
- CNN: Convolutional Neural Network - A class of deep neural networks, most commonly applied to analyzing visual imagery.
- VGG16: A deep convolutional neural network architecture named after the Visual Geometry Group at the University of Oxford, consisting of 16 convolutional and fully connected layers.

### Activation Functions
- ReLU (Rectified Linear Unit): An activation function commonly used in neural networks, defined as max(0, x). It introduces non-linearity to the network and helps in learning complex patterns.
- Softmax Activation: An activation function used in the output layer of a classification model. It converts raw scores into probabilities, with each score representing the likelihood of a class.

### Regularization Techniques
- Dropout: A regularization technique used to prevent overfitting in neural networks. It randomly drops a proportion of neurons during training to reduce interdependence among them.

### Layers in Convolutional Neural Networks
- Conv2D: Convolutional layer in a CNN. It performs convolutional operation on input data with a set of filters to extract features.
- MaxPooling2D: Pooling layer in a CNN. It reduces the spatial dimensions of the input volume by taking the maximum value over a pool of neighboring pixels.
- Flatten: Layer in a CNN that flattens the input into a 1D array, typically used before feeding the data into fully connected layers.
- Dense: Fully connected layer in a neural network. Each neuron in this layer is connected to every neuron in the previous layer.

### Data Augmentation Techniques
- Rescale: Normalizes the pixel values of the images to a range of [0, 1].
- Shear Range: Randomly applies shear transformations to the images.
- Rotation Range: Randomly rotates the images by a specified range of degrees.
- Zoom Range: Randomly zooms into the images by a specified range.
- Horizontal Flip: Randomly flips the images horizontally.
- Vertical Flip: Randomly flips the images vertically.

## Models
### CNN Model
![Simple CNN Model Results](/images/m1_model.png)

### VGG16-Inspired CNN Model
![Simple CNN Model Results](/images/m2_model.png)


## Results

### CNN Model
#### CNN Model
![CNN Model Accuracay](/images/m1_acc.png)
![CNN Model Loss](/images/m1_loss.png)

