import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import tensorflow as tf

from keras import layers, Sequential, callbacks

import warnings
warnings.filterwarnings("ignore")


# Define the image size and batch size
IMG_SIZE = (96, 96)
BATCH_SIZE = 64


def get_data_generators(train_dir, test_dir):
    """
    Generates data batches for training and testing.
    :param train_dir: Path to the directory containing training images.
    :param test_dir: Path to the directory containing testing images.
    :return: A tuple containing the training and testing data generators.
    """
    # Data augmentation for training images
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # No data augmentation for testing images
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)

    # Flow training images in batches using train_datagen generator
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary'
    )

    # Flow testing images in batches using test_datagen generator
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary'
    )

    return train_generator, test_generator


def build_model(input_shape=IMG_SIZE + (3,)):
    """
    Builds a convolutional neural network model.
    :param input_shape: Shape of the input images.

    :return: keras.Sequential: The compiled CNN model.
    """
    # Build a convolutional neural network model
    model = Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    return model


def train_model(model, train_generator, test_generator, epochs=10):
    """
    trains the CNN model
    :param:
    - model (keras.Sequential): Compiled CNN model.
    - train_generator (DirectoryIterator): Data generator for training images.
    - test_generator (DirectoryIterator): Data generator for testing images.
    - epochs (int): Number of epochs for training.
    :return: History: Training history.
    """
    # Train the model on data generated batch-by-batch by the generator
    history = model.fit(
        train_generator,
        steps_per_epoch=len(train_generator) // BATCH_SIZE,
        epochs=epochs,
        validation_data=test_generator,
        validation_steps=len(test_generator) // BATCH_SIZE
    )

    return history


def plot_training_results(history):
    """
    Plots training and validation accuracy and loss.
    :param history: Training history returned by model.fit().
    """
    # Plot training and validation accuracy and loss
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()


# Function to plot the ROC curve
def plot_roc_curve(y_true, y_score):
    """
    Plots the ROC curve.
    :param y_true: True binary labels.
    :param y_score: Target scores, can either be probability estimates
    or confidence values.
    :return: the plot
    """
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()


if __name__ == "__main__":
    # Path to train and test directories
    train_dir = '/Users/ozlemseyrani/outputFolder/train'
    test_dir = '/Users/ozlemseyrani/outputFolder/test'
    model_dir = '/Users/ozlemseyrani/trainedModels'

    # Get data generators for train and test sets
    train_generator, test_generator = get_data_generators(train_dir,
                                                          test_dir)
    # Build CNN model
    model = build_model()

    # Save the model to disk
    tf.saved_model.save(model, model_dir)

    # Train the model
    history = train_model(model, train_generator, test_generator)
    # Get test accuracy
    test_loss, test_accuracy = model.evaluate(test_generator)

    print("Test Loss:", test_loss)
    print("Test Accuracy:", test_accuracy)

    # Plot training results
    plot_training_results(history)

    # Make predictions on the test set
    predictions = model.predict(test_generator)

    # Get true labels for the test set
    y_true = test_generator.classes

    # Make predictions on the test set
    y_score = model.predict(test_generator).ravel()

    # Plot ROC curve
    plot_roc_curve(y_true, y_score)

