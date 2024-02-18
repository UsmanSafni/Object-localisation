# Object Localization using TensorFlow

## Introduction

The project focuses on Object Localization using TensorFlow, a popular deep learning framework. Object localization involves training a model to identify and locate objects within an image. TensorFlow is utilized for building and training the model.

## Data Overview

- **Dataset:** The dataset used consists of emojis, each associated with a unique class.
- **Visualization:** Emojis are visualized through images, and the goal is to train a model to accurately classify and locate these emojis within images.
- **Emoji Classes:** The dataset includes emojis such as happy, laughing, skeptical, sad, cool, whoa, crying, puking, and nervous.

## Problem Definition

The main problem addressed in this project is object localization. Given an input image, the model should be able to classify the emoji and accurately predict its location within the image. The problem involves both classification (identifying the emoji class) and regression (predicting the bounding box coordinates).

## Methodology

### Data Preparation

- The project starts by creating examples, generating bounding boxes, and plotting bounding boxes on images.
- Data generators are implemented to yield batches of training data.

### Model Architecture

- A Convolutional Neural Network (CNN) is designed using the Keras API.
- The CNN includes convolutional layers, batch normalization, max-pooling, and dense layers for both classification and bounding box regression.

### Custom Metrics and Callbacks

- A custom IoU (Intersection over Union) metric is implemented to evaluate the model's performance in terms of object localization.
- A custom callback for learning rate scheduling is used during model training.

### Model Compilation and Training

- The model is compiled with categorical crossentropy loss for classification and custom IoU for bounding box regression.
- The Adam optimizer is employed with a specific learning rate.
- The model is trained for a specified number of epochs with the defined data generator and callbacks.

## Results

The results section evaluates the trained model's performance using the custom IoU metric and categorical crossentropy. The model's ability to classify emojis and accurately localize them within images is assessed. Test cases are presented, showcasing predictions against ground truth annotations. Visualization tools, such as matplotlib, are utilized to display images with predicted bounding boxes. The total loss for this model was nearly 0.003, owing to loss from classification and regression, and the accuracy of this model is nearly 100%.

## Conclusion

The project demonstrates the end-to-end process of creating and training a TensorFlow model for object localization. The model architecture and custom metrics aim to achieve accurate and reliable predictions for emoji classification and bounding box regression. The methodology provides a detailed walkthrough of the steps involved, and the results section summarizes the model's performance.

