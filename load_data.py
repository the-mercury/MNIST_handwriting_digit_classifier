import gzip
import numpy as np
import tensorflow as tf


def __data_images(data_path):
    """This method loads the images in MNIST dataset"""

    with gzip.open(data_path, 'r') as f:
        # first 4 bytes is a magic number
        f.seek(4)
        # second 4 bytes is the number of images
        image_count = int.from_bytes(f.read(4), 'big')
        # third 4 bytes is the row count
        row_count = int.from_bytes(f.read(4), 'big')
        # fourth 4 bytes is the column count
        column_count = int.from_bytes(f.read(4), 'big')
        image_data = f.read()
        images = np.frombuffer(image_data, dtype=np.uint8).reshape(image_count, row_count, column_count, 1)

        return images / 255.0, images.shape


def __data_labels(label_path):
    """This method loads the labels from MNIST dataset"""

    with gzip.open(label_path, 'r') as f:
        # first 4 bytes is a magic number
        f.seek(4)
        # second 4 bytes is the number of labels
        row_count = int.from_bytes(f.read(4), 'big')
        label_data = f.read()
        labels = np.frombuffer(label_data, dtype=np.uint8).reshape((row_count, 1))

        return labels, labels.shape


def create_dataset(data_path, label_path):
    """This method creates dataset using np arrays generated from reading byte_data_file"""

    (data_array, input_shape) = __data_images(data_path)
    # data_array = tf.cast(data_images(data_path) / 255, tf.float32)
    (label_array, output_shape) = __data_labels(label_path)
    # label_array = tf.cast(data_labels(label_path), tf.float32)
    dataset_tensor = tf.data.Dataset.from_tensor_slices((data_array, label_array))
    print(dataset_tensor)

    return dataset_tensor, input_shape, output_shape


def batch_shuffled(train_dataset, shuffle_buffer_size, batch_size):
    """This method shuffles the data and return a batch given both shuffle buffer and batch size"""

    batch_shuffled_data = train_dataset.shuffle(shuffle_buffer_size).batch(batch_size)
    print(batch_shuffled_data)

    return batch_shuffled_data
