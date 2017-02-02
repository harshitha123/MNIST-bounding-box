from __future__ import division
import tensorflow as tf
import numpy as np


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(
        x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def train_model(lr, batch_X_train, X_validation,
                batch_bounds_train, bounds_validation):

    X_train = np.concatenate(batch_X_train)
    bounds_train = np.concatenate(batch_bounds_train)

    sess = tf.InteractiveSession()
    x = tf.placeholder(tf.float32, shape=[None, 28, 28])
    y = tf.placeholder(tf.float32, shape=[None, 4])

    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])

    x_image = tf.reshape(x, [-1, 28, 28, 1])

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([1024, 4])
    b_fc2 = bias_variable([4])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    x_inner_lower = tf.maximum(y_conv[:, 0], y[:, 0])
    x_inner_upper = tf.minimum(y_conv[:, 1], y[:, 1])
    y_inner_lower = tf.maximum(y_conv[:, 2], y[:, 2])
    y_inner_upper = tf.minimum(y_conv[:, 3], y[:, 3])

    area_inner = tf.multiply(
        tf.subtract(x_inner_upper, x_inner_lower) + 1,
        tf.subtract(y_inner_upper, y_inner_lower) + 1,
    )
    area_outer1 = tf.multiply(
        tf.subtract(y_conv[:, 1], y_conv[:, 0]) + 1,
        tf.subtract(y_conv[:, 3], y_conv[:, 2]) + 1,
    )
    area_outer2 = tf.multiply(
        tf.subtract(y[:, 1], y[:, 0]) + 1,
        tf.subtract(y[:, 3], y[:, 2]) + 1,
    )

    iou = tf.reduce_mean(tf.divide(
        area_inner, tf.add(area_outer1, area_outer2 - area_inner)))

    train_step = tf.train.AdamOptimizer(lr).minimize(-1 * iou)
    accuracy = iou
    sess.run(tf.global_variables_initializer())
    for i in range(len(batch_X_train)):
        if i % 50 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                x: batch_X_train[i],
                y: batch_bounds_train[i],
                keep_prob: 1.,
            })
            validation_accuracy = accuracy.eval(feed_dict={
                x: X_validation, y: bounds_validation, keep_prob: 1.
            })
            print("Step {}, training accuracy {:.4f}, validation accuracy: "
                  "{:.4f}".format(i, train_accuracy, validation_accuracy))
        train_step.run(feed_dict={
            x: batch_X_train[i],
            y: batch_bounds_train[i],
            keep_prob: 0.5,
        })

    print("Training accuracy {:.4f}".format(accuracy.eval(feed_dict={
        x: X_train, y: bounds_train, keep_prob: 1.0})))
    print("Validation accuracy {:.4f}".format(accuracy.eval(feed_dict={
        x: X_validation, y: bounds_validation, keep_prob: 1.0})))

    pred_validation = y.eval(feed_dict={
        x: X_validation, y: bounds_validation, keep_prob: 1.0})

    return pred_validation
