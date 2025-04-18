# Image Classification with CNN and Grad-CAM Visualization

This project demonstrates the use of Convolutional Neural Networks (CNNs) for image classification, where images are categorized into two classes—normal and anomalous. The project also incorporates Grad-CAM (Gradient-weighted Class Activation Mapping) to visualize which parts of an image the model focuses on when making its predictions, offering insights into the decision-making process.

## Project Overview

- **Model Architecture**: The model is built using a custom CNN architecture with multiple convolutional layers, max-pooling, and fully connected layers.
- **Visualization**: Grad-CAM is applied to highlight the important regions of the image that the model uses for classification.
- **Evaluation**: The model's performance is evaluated using Mean Absolute Error (MAE) and a Confusion Matrix.
- **Scatter Plot**: A scatter plot is used to show the relationship between true and predicted anomaly scores for both training and test datasets.

## Features

### Data Preprocessing:
- Loads and processes images into two categories: normal and anomalous.
- Converts labels into one-hot encoded vectors.

### Model Training:
- The CNN model is trained on the preprocessed image data with 250 epochs.
- Evaluates the model's performance using the MAE metric and Confusion Matrix.

### Grad-CAM Visualization:
- Generates a heatmap that highlights the regions of the input images that the model considers most important.
- Overlays the heatmap on the original image to visualize areas of importance.

### Evaluation:
- Mean Absolute Error (MAE) measures the accuracy of the predicted anomaly scores compared to the true values.
- Confusion Matrix shows how well the model classifies normal vs anomalous images.
- Scatter Plot shows the relationship between the true labels and predicted anomaly scores.



### Image Dimensions
- Image Size: 325x325 pixels
- Image Format: RGB (3 channels)

## Model Architecture

The model is a simple CNN built with the following layers:

### Convolutional Layers:
- 3 convolutional layers with 5 filters each.
- Each convolution is followed by max-pooling layers.

### Fully Connected Layers:
- Dense layers with 100, 30, and 20 neurons.
- The final output layer has a single neuron using a linear activation function to predict the anomaly score.

## Training

The model is trained for 250 epochs using the RMSprop optimizer with a learning rate of 10^(-3). The loss function used is Mean Absolute Error (MAE), which is appropriate for regression tasks where the goal is to predict continuous values (anomaly score).

## Usage

### 1. Loading and Preprocessing Data
The images are loaded from the directory and processed as follows:
- Images are normalized to the range [0, 1].
- Labels are converted into one-hot encoded vectors.
- The dataset is split into training and test sets.

### 2. Training the Model
The CNN model is trained using the following steps:
- Model Compilation: The model uses the RMSprop optimizer and MAE loss function.
- Model Fitting: The model is trained for 250 epochs with a batch size of 8.

### 3. Evaluating the Model
After training, the model is evaluated using Mean Absolute Error (MAE) and a Confusion Matrix.
- MAE: The MAE for training is 0.51, and for the test data, it is 0.49. These values indicate the average difference between the predicted and actual anomaly scores.
- Confusion Matrix: The confusion matrix provides insights into how well the model classifies normal and anomalous images.

### 4. Grad-CAM Visualization
Grad-CAM is used to generate a heatmap that highlights the important regions of the input image that the model focused on for its prediction. The heatmap is overlaid on the original image to visually show which areas were most influential for the classification.

### 5. Visualizing the Results
After the predictions, several visualizations are generated:
- Input Image: Displays the original image.
- Grad-CAM Heatmap: A heatmap showing which areas of the image are important for the classification.
- Predicted Segments: The model's prediction overlaid on the original image, with Grad-CAM highlighting the relevant regions.

### 6. Scatter Plot
The scatter plot shows the relationship between the true and predicted anomaly scores for both training and test datasets.
- Training Data: A scatter plot comparing the true vs predicted values for training data.
- Test Data: A scatter plot comparing the true vs predicted values for test data.

## Example Visualization

### Scatter Plot of Predictions:
This plot shows the relationship between the true and predicted values. The red line represents the perfect prediction (where true values match predicted values). The model's predictions for the test set are scattered around this line.

### Model Prediction on Test Set:
This image shows the predictions of the model overlaid on the test set images, with each predicted anomaly score indicated on the image.

## Conclusion

This project demonstrates how Convolutional Neural Networks (CNNs) can be used for image classification tasks, focusing on detecting anomalies. Grad-CAM is utilized to interpret and visualize the model's decision-making process, offering insights into which parts of the image the model focuses on for classification. The model's performance is evaluated using Mean Absolute Error (MAE) and Confusion Matrix, and visualizations like scatter plots and Grad-CAM heatmaps further help in understanding and improving the model.