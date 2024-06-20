# Image Classification using Support Vector Machine (SVM)

This project implements a Support Vector Machine (SVM) algorithm to classify images of cats and dogs from the Kaggle dataset. The dataset used for this project is sourced from the [Dogs vs. Cats](https://www.kaggle.com/c/dogs-vs-cats/data) competition on Kaggle.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model](#model)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction
Image classification is a fundamental task in computer vision, and Support Vector Machine (SVM) is a popular machine learning algorithm used for this purpose. This project aims to classify images of cats and dogs using an SVM classifier.

## Dataset
The dataset used in this project contains a large number of images of cats and dogs. The goal is to train an SVM classifier to distinguish between images of cats and dogs accurately.

You can download the dataset from [here](https://www.kaggle.com/c/dogs-vs-cats/data).

## Installation
To run this project, you need to have Python installed along with the following libraries:
- numpy
- pandas
- scikit-learn
- matplotlib
- OpenCV (cv2)

You can install the required libraries using pip:

pip install numpy pandas scikit-learn matplotlib opencv-python

<h1><b>Usage</b></h1>

**Clone the repository:**

git clone https://github.com/your-username/image-classification-svm.git

**Navigate to the project directory:**

cd image-classification-svm

**Run the svm_image_classification.py script to train the SVM model and classify images:**

python svm_image_classification.py

<h1><b>Model</b></h1>

The Support Vector Machine (SVM) classifier is implemented using the scikit-learn library. The key steps involved are:

**Data Preprocessing:** Preprocessing the images (e.g., resizing, converting to grayscale) and extracting features (e.g., HOG features).

**Model Training:** Training an SVM classifier on the feature vectors extracted from the images.

**Model Evaluation:** Evaluating the performance of the SVM classifier using metrics such as accuracy and confusion matrix.

<h1><b>Results</b></h1>

The performance of the SVM classifier is evaluated based on its accuracy in classifying images of cats and dogs. Visualizations such as confusion matrix and classification reports are used to analyze the results.
