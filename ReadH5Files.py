import os
import h5py
from PIL import Image
import numpy as np


def split_and_save_images(train_x_file, train_y_file, test_x_file, test_y_file, output_folder):
    """
    Splits images and labels from H5 files and saves them into train and test folders.

    :param:
    - train_x_file (str): Path to the H5 file containing training images.
    - train_y_file (str): Path to the H5 file containing training labels.
    - test_x_file (str): Path to the H5 file containing testing images.
    - test_y_file (str): Path to the H5 file containing testing labels.
    - output_folder (str): Path to the output folder where the images will be saved.
    """
    # Open the H5 files
    train_x_h5 = h5py.File(train_x_file, 'r')
    train_y_h5 = h5py.File(train_y_file, 'r')
    test_x_h5 = h5py.File(test_x_file, 'r')
    test_y_h5 = h5py.File(test_y_file, 'r')

    # Extracting images and labels from H5 files
    train_images = np.array(train_x_h5['x'])
    train_labels = np.array(train_y_h5['y']).flatten()
    test_images = np.array(test_x_h5['x'])
    test_labels = np.array(test_y_h5['y']).flatten()

    # Create output folders if they don't exist
    train_folder = os.path.join(output_folder, 'train')
    test_folder = os.path.join(output_folder, 'test')
    for folder in [train_folder, test_folder]:
        os.makedirs(os.path.join(folder, 'YesCancer'), exist_ok=True)
        os.makedirs(os.path.join(folder, 'NoCancer'), exist_ok=True)

    # # Save train images
    for i, (image, label) in enumerate(zip(train_images, train_labels)):
        folder = 'YesCancer' if label == 1 else 'NoCancer'
        image = Image.fromarray(image)
        image_path = os.path.join(train_folder, folder, f"image_{i}.png")
        image.save(image_path)

    # Save test images
    for i, (image, label) in enumerate(zip(test_images, test_labels)):
        folder = 'YesCancer' if label == 1 else 'NoCancer'
        image = Image.fromarray(image)
        image_path = os.path.join(test_folder, folder, f"image_{i}.png")
        image.save(image_path)

    # Close the H5 files
    # train_x_h5.close()
    # train_y_h5.close()
    test_x_h5.close()
    test_y_h5.close()

# Example usage
output_folder = '/Users/ozlemseyrani/outputFolder'
split_and_save_images('camelyonpatch_level_2_split_train_x.h5', 'camelyonpatch_level_2_split_train_y.h5',
                      'camelyonpatch_level_2_split_test_x.h5', 'camelyonpatch_level_2_split_test_y.h5',
                      output_folder)
