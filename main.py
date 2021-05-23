import os
import load_data
import nn_model
import tensorflow as tf
import datetime
from tensorboard import program


DIR_NAME = os.path.dirname(__file__)
TRAIN_DATA_PATH = os.path.join(DIR_NAME, 'MNIST_handwriting_digits', 'train', 'train-images-idx3-ubyte.gz')
TRAIN_LABEL_PATH = os.path.join(DIR_NAME, 'MNIST_handwriting_digits', 'train', 'train-labels-idx1-ubyte.gz')
TEST_DATA_PATH = os.path.join(DIR_NAME, 'MNIST_handwriting_digits', 'test', 't10k-images-idx3-ubyte.gz')
TEST_LABEL_PATH = os.path.join(DIR_NAME, 'MNIST_handwriting_digits', 'test', 't10k-labels-idx1-ubyte.gz')
BATCH_SIZE = 50
SHUFFLE_BUFFER_SIZE = 2048
OUTPUT_LAYER_NUM = 10
EPOCH_NUM = 10


def main():
    """This is the main method,
    loading MNIST dataset from directory and run a CNN to train the model
    and recognize handwriting digits"""

    (train_dataset, train_data_shape, train_label_shape) = load_data.create_dataset(TRAIN_DATA_PATH, TRAIN_LABEL_PATH)
    print('Train data shape: ', train_data_shape)
    print('Train label shape: ', train_label_shape)
    (test_dataset, test_data_shape, test_label_shape) = load_data.create_dataset(TEST_DATA_PATH, TEST_LABEL_PATH)
    print('Test data shape: ', test_data_shape)
    print('Test label shape: ', test_label_shape)
    train_dataset_shuffled_batch = load_data.batch_shuffled(train_dataset, SHUFFLE_BUFFER_SIZE, BATCH_SIZE)
    test_dataset_batch = test_dataset.batch(BATCH_SIZE)

    model = nn_model.design_model(train_data_shape[1:], OUTPUT_LAYER_NUM)
    nn_model.compile_model(model)
    model.summary()
    log_dir = 'logs\\fit\\' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    nn_model.fit_eval(model,
                      train_dataset_shuffled_batch,
                      test_dataset_batch,
                      EPOCH_NUM,
                      DIR_NAME,
                      tensorboard_callback)
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', log_dir])
    tb.main()


if __name__ == '__main__':
    main()
