import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

import os
import pickle

def load_cifar10_batch(file):
    with open(file, 'rb') as f:
        data_dict = pickle.load(f, encoding='bytes')
        images = data_dict[b'data']
        labels = data_dict[b'labels']
        images = images.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
        labels = np.array(labels)
    return images, labels

def load_cifar10(data_dir):
    train_files = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
    test_file = 'test_batch'
    
    x_train, y_train = [], []
    for file in train_files:
        file_path = os.path.join(data_dir, file)
        images, labels = load_cifar10_batch(file_path)
        x_train.append(images)
        y_train.append(labels)
    
    x_train = np.concatenate(x_train)
    y_train = np.concatenate(y_train)
    
    x_test, y_test = load_cifar10_batch(os.path.join(data_dir, test_file))
    
    return (x_train, y_train), (x_test, y_test)


# Load the CIFAR-10 dataset
data_dir = 'cifar-10-batches-py'
(x_train, y_train), (x_test, y_test) = load_cifar10(data_dir)

# Preprocess the data
x_train = x_train.reshape(-1, 32 * 32 * 3) / 255.0
x_test = x_test.reshape(-1, 32 * 32 * 3) / 255.0
y_train = y_train.ravel()
y_test = y_test.ravel()

# Create validation set
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

# Use a smaller subset of the dataset for k-NN and SVM
sample_size = 10000
x_train_small = x_train[:sample_size]
y_train_small = y_train[:sample_size]
x_val_small = x_val[:sample_size]
y_val_small = y_val[:sample_size]

# PCA for dimensionality reduction
# pca = PCA(n_components=50)
# x_train_pca = pca.fit_transform(x_train_small)
# x_val_pca = pca.transform(x_val_small)

# # K-Nearest Neighbors
# k_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# knn_accuracies = []
# max_knn_accuracy = 0

# for k in k_values:
#     print(k)
#     knn = KNeighborsClassifier(n_neighbors=k)
#     knn.fit(x_train_pca, y_train_small)
#     y_pred_knn = knn.predict(x_val_pca)
#     knn_accuracy = accuracy_score(y_val_small, y_pred_knn)
#     knn_accuracies.append(knn_accuracy)
#     max_knn_accuracy = max(max_knn_accuracy, knn_accuracy)

# # Print max KNN accuracy
# print("Max KNN accuracy:", max_knn_accuracy)

# # Plot the results for K-Nearest Neighbors
# plt.plot(k_values, knn_accuracies, marker='o')
# plt.xlabel('k (number of neighbors)')
# plt.ylabel('Accuracy')
# plt.title('k-NN Accuracy vs. Number of Neighbors')
# plt.show()

# # Support Vector Machine
# C_values = np.linspace(1, 10, 5)
# gamma_values = np.linspace(0.001, 0.2, 5)

# svm_accuracies = np.zeros((len(C_values), len(gamma_values)))
# max_svm_accuracy = 0

# for i, C in enumerate(C_values):
#     for j, gamma in enumerate(gamma_values):
#         print(C, gamma)
#         svm = SVC(kernel='rbf', C=C, gamma=gamma)
#         svm.fit(x_train_pca, y_train_small)
#         y_pred_svm = svm.predict(x_val_pca)
#         svm_accuracy = accuracy_score(y_val_small, y_pred_svm)
#         svm_accuracies[i, j] = svm_accuracy
#         max_svm_accuracy = max(max_svm_accuracy, svm_accuracy)

# # Print max SVM accuracy
# print("Max SVM accuracy:", max_svm_accuracy)

# # Plot the results for Support Vector Machine
# plt.imshow(svm_accuracies, cmap='viridis', aspect='auto', origin='lower',
#            extent=[gamma_values[0], gamma_values[-1], C_values[0], C_values[-1]])
# plt.colorbar(label='Accuracy')
# plt.xlabel('Gamma')
# plt.ylabel('C (Regularization parameter)')
# plt.title('SVM Accuracy vs. Regularization Parameter and Gamma')
# plt.show()


# Function to create a CNN model with a given number of convolutional layers
def create_cnn_model(num_conv_layers):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(32, 32, 3)))
    
    for _ in range(num_conv_layers):
        model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu'))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(512, activation='relu'))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    
    return model

# Define the number of convolutional layers and epochs to test
num_conv_layers_list = [1, 2]
epochs_list = [5, 10, 15]

# Initialize arrays to store results
accuracies = np.zeros((len(num_conv_layers_list), len(epochs_list)))

# Loop through the combinations of convolutional layers and epochs
for i, num_conv_layers in enumerate(num_conv_layers_list):
    for j, epochs in enumerate(epochs_list):
        cnn_model = create_cnn_model(num_conv_layers)
        cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        cnn_model.fit(x_train.reshape(-1, 32, 32, 3), y_train, epochs=epochs, validation_data=(x_val.reshape(-1, 32, 32, 3), y_val))
        cnn_accuracy = cnn_model.evaluate(x_val.reshape(-1, 32, 32, 3), y_val)[1]
        accuracies[i, j] = cnn_accuracy

# Plot the results
fig, ax = plt.subplots()
cax = ax.imshow(accuracies, cmap='viridis', aspect='auto', origin='lower',
                extent=[epochs_list[0], epochs_list[-1], num_conv_layers_list[0], num_conv_layers_list[-1]])
plt.colorbar(cax, label='Accuracy')
plt.xlabel('Number of Epochs')
plt.ylabel('Number of Convolutional Layers')
plt.title('CNN Accuracy vs. Number of Convolutional Layers and Epochs')
plt.show()


# # Compare the results
# results = pd.DataFrame({
#     'Algorithm': ['k-NN', 'SVM', 'CNN 1'],
#     'Accuracy': [max_knn_accuracy, max_svm_accuracy, cnn_accuracy]
# })

# print(results)