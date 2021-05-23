import tensorflow as tf
import numpy as np
from tensorboard import program
from tensorflow.keras.utils import plot_model


image = tf.keras.preprocessing.image.load_img(r'my_testset\test_3.png', color_mode='grayscale', target_size=(28, 28, 1))
input_arr = tf.keras.preprocessing.image.img_to_array(image)
input_arr = np.array([input_arr]) / 255  # Normalize & convert a single image to a batch.
input_arr = 1 - input_arr  # Inverting the image to match training dataset
print(input_arr.shape)

model = tf.keras.models.load_model('saved_model')
# plot_model(model, to_file='model_plot', show_shapes=True, show_layer_names=True)
model.summary()

log_dir = 'logs\\fit'
tb = program.TensorBoard()
tb.configure(argv=[None, '--logdir', log_dir])
tb.main()

prediction = model.predict(input_arr)
print('\nmaximum probability:', np.max(prediction))
print('\nthe probability array of the input image:\n', prediction)
print('\n>>>the predicted number is:', np.argmax(prediction))

