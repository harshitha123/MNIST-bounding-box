import numpy as np
import matplotlib.pyplot as plt


def section_print():
    '''Memorized function keeping track of section number'''
    section_number = 0

    def __inner(message):
        nonlocal section_number
        section_number += 1
        print('Section {}: {}'.format(section_number, message))
    print('Section {}: Initializing section function'.format(section_number))
    return __inner


def get_bounding_box(grad, threshold):
    """Get the bounding box around a digit, expressed as x and y coordinates
    """
    exceeds_threshold = np.absolute(grad) > threshold
    diff = np.diff(exceeds_threshold)
    boundaries = []
    for i, (e, d) in enumerate(zip(exceeds_threshold, diff)):
        breaks = np.where(d)[0]
        assert breaks.shape[0] > 0
        if e[0]:
            breaks = np.array([0, breaks[0]])
        if e[-1]:
            breaks = np.array([breaks[0], d.shape[0]])
            breaks
        breaks[0] = breaks[0] + 1
        boundary = (breaks[0], breaks[-1])
        boundaries.append(boundary)
    return np.array(boundaries)


def get_data_to_box(df, threshold=.1):
    """get bounding boxes for all digits in dataset df
    """
    z_grad, y_grad, x_grad = np.gradient(df)
    y_grad_1d = y_grad.sum(axis=2)
    x_grad_1d = x_grad.sum(axis=1)

    y_bounds = get_bounding_box(y_grad_1d, threshold)
    x_bounds = get_bounding_box(x_grad_1d, threshold)

    return np.hstack([x_bounds, y_bounds])


def plot_bounding_box(data, bounds, index):
    """Plot the image and bounding box indexed by index
    """

    # Calculate bounding box
    print(index)
    x_bound = bounds[index, 0:2] * data.shape[1]
    y_bound = bounds[index, 2:4] * data.shape[2]

    # Plot image
    img = plt.imshow(data[index], cmap='Greys')

    # Plot calculated bounding box
    plt.plot([x_bound[0]] * 2, y_bound, color='r')
    plt.plot(x_bound, [y_bound[0]] * 2, color='r')
    plt.plot([x_bound[1]] * 2, y_bound, color='r')
    plt.plot(x_bound, [y_bound[1]] * 2, color='r')

    return img


def plot_bounding_grid(df, subplot_shape, bounding_boxes):
    fig, axes = plt.subplots(*subplot_shape)
    for ax in axes.ravel():
        rand_index = np.random.randint(0, df.shape[0])
        ax.imshow(df[rand_index], cmap='Greys')
        ax.axis('off')

        # Calculate bounding box
        x_bound = bounding_boxes[rand_index, 0:2] * df.shape[1]
        y_bound = bounding_boxes[rand_index, 2:4] * df.shape[2]

        # Plot calculated bounding box
        ax.plot([x_bound[0]] * 2, y_bound, color='r')
        ax.plot(x_bound, [y_bound[0]] * 2, color='r')
        ax.plot([x_bound[1]] * 2, y_bound, color='r')
        ax.plot(x_bound, [y_bound[1]] * 2, color='r')

    fig.tight_layout()
    fig.subplots_adjust(wspace=-.1, hspace=-.1)
    return fig


def reshape_to_img(df):
    """Reshape 1d images to 2d
    """
    m = df.shape[0]
    i = np.int(np.sqrt(df.shape[1]))
    return df.reshape((m, i, i))


def get_bounding_boxes(mnist):
    """Get bounding boxes for the MNIST train, validation and test sets
    """
    train = reshape_to_img(mnist.train.images)
    validation = reshape_to_img(mnist.validation.images)
    test = reshape_to_img(mnist.test.images)

    bounds_train = get_data_to_box(train) * 1. / 28
    bounds_validation = get_data_to_box(validation) * 1. / 28
    bounds_test = get_data_to_box(test) * 1. / 28

    return bounds_train, bounds_validation, bounds_test


def add_background_noise(df, treshold=.01, scale=1.):
    noise_mask = df < treshold
    noise_overlay = np.divide(
        noise_mask, (1 + np.exp(np.random.normal(size=df.shape))))
    return df + noise_overlay
