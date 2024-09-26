# Image-classification-with-Cnn

## Overview
This project aims to automatically categorize products based on their images using a deep learning model. The dataset used in this project is provided by Torob, a shopping search engine that aggregates product information from various online stores. By doing so, users can easily search for products and compare different sellers in a single interface.
The key objective of this project is to build a Convolutional Neural Network (CNN) model capable of classifying products into appropriate categories based on product images. In a platform like Torob, accurate product categorization is crucial for improving the searchability and comparison of products, making it easier for users to find what they are looking for.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Data Description](#data-description)
- [Contributing](#contributing)

## Prerequisites

Before you begin, ensure you have the following installed:

- Python 3.x
- Jupyter Notebook or any Python IDE
- Required libraries:
  - Pandas
  - NumPy
  - Keras
  - Matplotlib
  - TensorFlow

You can install these libraries using pip if they are not already installed.

## Installation

1. Clone the Repository: Start by cloning this repository to your local machine.

2. Download the Dataset: The project includes a step to download the dataset from Google Drive. Ensure you have access to the dataset and that the link is valid.

3. Unzip the Dataset: After downloading, the dataset will be in a zip format. The project includes a command to unzip this file, making the data accessible for further processing.

## Model Architecture 

The project leverages Convolutional Neural Networks (CNNs), a class of deep neural networks, particularly well-suited for image classification tasks. The following steps are taken to design the model:

Data Preprocessing: Preprocessing the dataset to normalize images and prepare them for input into the CNN model.
Constructing Model (Using Transfer Learning): Using transfer learing in order to take advantage of the ResNet model arcitecture with self-built prediction head (FNN head).
Evaluation metrics and performance analysis: The model is evaluated based on accuracy score to ensure the performance meets the project requirements.

## Usage

1. Environment Setup: The code configures Keras to use TensorFlow as its backend, which is essential for compatibility with various functionalities provided by Keras.

2. Data Preparation: The dataset is loaded and split into training and validation sets. Images are resized, and labels are inferred from the directory structure.

3. Data Augmentation: To improve model robustness, data augmentation techniques such as random zoom and brightness adjustments are applied to the training dataset.

4. Model Building: A pre-trained ResNet50 model is utilized as a base, with additional layers added for classification tasks. The model is configured to be trained on the specified number of classes.

5. Model Training: The model is compiled and trained using the training dataset, with validation metrics monitored to prevent overfitting.

6. Evaluation: After training, the model's performance is evaluated on the validation dataset, providing insights into its accuracy.

## Data Description

The dataset used in this project consists of images organized into subdirectories based on their respective classes. Each subdirectory contains images belonging to that class, which allows for easy inference of labels during data loading.