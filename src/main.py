from os import path
import time

import config
import utils
import model

import tensorflow as tf
sess = tf.InteractiveSession()
from tensorflow.examples.tutorials.mnist import input_data

____section____ = utils.section_print()
mnist = input_data.read_data_sets(
    path.join(config.path_main, 'MNIST_data'), one_hot=True)

____section____('Generate synthetic bounding boxes')

train = utils.reshape_to_img(mnist.train.images)
validation = utils.reshape_to_img(mnist.validation.images)
test = utils.reshape_to_img(mnist.test.images)

bounds_train = utils.get_data_to_box(train) * 1. / 28
bounds_validation = utils.get_data_to_box(validation) * 1. / 28
bounds_test = utils.get_data_to_box(test) * 1. / 28

bounding_box_grid_generated = utils.plot_bounding_grid(
    df=train,
    subplot_shape=(4, 6),
    bounding_boxes=bounds_train,
)
bounding_box_grid_generated.savefig(
    path.join(config.path_outputs, 'bounding_box_grid_generated.png'))


____section____('Model bounding box')
start = time.time()
pred_validation = model.train_model(mnist, bounds_train, bounds_validation)
print('Training completed in {:.0f} seconds'.format(time.time() - start))

bounding_box_grid_estimated = utils.plot_bounding_grid(
    df=validation,
    subplot_shape=(4, 6),
    bounding_boxes=pred_validation,
)
bounding_box_grid_estimated.savefig(
    path.join(config.path_outputs, 'bounding_box_grid_estimated.png'))
