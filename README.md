# Concrete-Crack-Image-Classification-Using-CNN
 
### Introduction
This is a model to identify the cracks on concrete images by using convolutional neural network. The datasets have 400000 images and been divided to 2 categories : Positive(Cracks) & Negative(Not Cracks). 
 
 OBJECTIVE : To perform image classification to classify concretes that with cracks or without cracks.
 
### Methodology

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)

For input layer, the model was expecting to get colour image with (180,180,3) dimensions with random rotation & random flip images by through the process of data augmentation.

Next, Transfer Learning approach was used in this model. Where the Feature Extractor, a pretrained model MobileNet_V2 used to pretrained the image datasets and the module was link with Tensorflow Keras API using Functional API method. 

A global average pooling and dense layer are used as the classifier to output softmax activation function. The softmax are used to predict the class of the input images.

The model architecture are shown in image below:

![model_list](https://user-images.githubusercontent.com/105650253/211811278-29876914-6f5b-4bb5-806e-a8470036d8b1.PNG)

![model architecture](https://user-images.githubusercontent.com/105650253/211809198-0ad5f0f6-9a03-42ec-a70f-170ef48626df.png)

Since it's a binary classification, the loss function used is sparse cross-entropy. The optimizer used is Adam.
The model is trained by using 5 number of epochs with 64 of batch_sizes. Early Stopping were used in this process to ensure the training process of the model are not overfitting. The training was done with 99% training accuracy and 99% validation accuracy.
![Tensorboard_Graph](https://user-images.githubusercontent.com/105650253/211811248-85fff989-33e9-41e1-97cc-bf63391fad98.PNG)



### Result
The model was evaluated with test data and shown 99% of accuracy.
![result](https://user-images.githubusercontent.com/105650253/211811573-cd2c0e34-3f4c-4f7e-8374-8d01574c89cb.PNG)

The model was deploy to shown the prediction vs actual result in images
![output result](https://user-images.githubusercontent.com/105650253/211813021-80a48838-4424-4f08-b7ea-b426571ad239.png)

### Discussion
The model was good enough to classify image classification, however there are some issues regarding on this model and can have some improvement:-
1. The pretrained model can be change to any applications model in Tensorflow documentation. The documentation can be refer at this website:https://www.tensorflow.org/tutorials/images/transfer_learning
2. Number of epochs can be increase if you have powerful internal harware of your machine so that the time can be reduced  and can applied EarlyStopping method to avoid the trained data are overfitting

### Acknowledgment
The dataset used in this analysis is from : https://data.mendeley.com/datasets/5y9wdsg2zt/2

