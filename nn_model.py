import os
import tensorflow as tf


def design_model(input_data_shape, output_layer_num):

    tf.keras.backend.set_floatx('float64')
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(12, 5, padding='same', activation='relu', input_shape=input_data_shape),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(24, 5, activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(32, 5, activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(output_layer_num, activation='softmax')
    ])

    return model


def compile_model(model):
    model.compile(optimizer='Adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])


def fit_eval(model, train_dataset_shuffled_batch, test_dataset_batch, epoch, dirname, tb_callback):
    fit_history = model.fit(train_dataset_shuffled_batch,
                            epochs=epoch,
                            validation_data=test_dataset_batch,
                            callbacks=[tb_callback],
                            verbose=2)
    os.makedirs(dirname, exist_ok=True)
    model.save('saved_model')
    return fit_history
