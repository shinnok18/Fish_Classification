# Fish Classification

This project focuses on classifying different types of fish using image data. The project involves data loading, preprocessing, model training using a Convolutional Neural Network (CNN), and performance evaluation.

## Project Structure

### 1. Importing The Modules
Essential Python libraries used include:
- TensorFlow/Keras for building and training the CNN model
- Matplotlib and Seaborn for visualizations
- Pandas for data manipulation

### 2. Loading The Fish Data
- The dataset contains fish images labeled by species. The data is loaded into a DataFrame for exploration.

#### a. Showing the First 25 Images
- A preview of the first 25 images in the dataset to gain insight into the data.

#### b. DataFrame Shape
- The dimensionality of the dataset is displayed for better understanding.

#### c. Types of Fishes
- Lists the unique fish species (labels) present in the dataset.

#### d. Graph of Image Counts
- A bar chart visualizes the distribution of images across different fish species.

### 3. Model Development

#### a. Splitting the Data
- The dataset is split into training and testing sets for model evaluation.

#### b. Data Augmentation
- To improve the model's robustness, data augmentation techniques like rotation, flipping, and zooming are applied to the training set.

#### c. Define and Compile the Model
- A Convolutional Neural Network (CNN) architecture is defined using Keras, with layers like Conv2D, MaxPooling, and Dense.
- The model is compiled using categorical cross-entropy as the loss function and Adam as the optimizer.

#### d. Training the Model
- The model is trained on the fish image dataset over multiple epochs, with validation data used for performance tracking.

#### e. Model Summary
- Displays a summary of the CNN architecture, including the number of layers and parameters.

#### f. Model Accuracy and Validation Performance
- Visualizes the accuracy and validation performance across epochs to assess how well the model generalizes.

#### g. Retrieve Training and Validation Metrics
- Metrics such as training accuracy, validation accuracy, loss, and validation loss are retrieved from the training history.

#### h. Model Testing: Prediction vs Actual
- The trained model is tested on the test set, and predictions are compared against the actual labels.

## Conclusion
This project successfully implements a CNN model for classifying fish species from images. The model is trained with data augmentation, and its performance is evaluated using accuracy metrics.

## Dependencies
- TensorFlow
- Keras
- Matplotlib
- Seaborn
- Pandas

## Kaggle Link
https://www.kaggle.com/code/oruntrkokulolu/fish-classification/
