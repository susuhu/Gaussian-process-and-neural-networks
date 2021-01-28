import time
import numpy as np
import matplotlib.pyplot as plt
import gpflow
from gpflow.utilities import print_summary, set_trainable

import tensorflow as tf
import tensorflow_probability as tfp

# Load and prepare the MNIST dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()  # 60k train, 10k test
x_train, x_test = x_train / 255.0, x_test / 255.0

# set numbers
NUM_TRAIN_DATA = np.array([60, 600, 6000, 60000])
NUM_TEST_DATA = x_test.shape[0]
MAXITER = 100
H = W = 28  # width and height
IMAGE_SHAPE = [H, W]
patch_shape = [5, 5]
num = 1
x_train = x_train[0:num]  # (n,28,28)
y_train = y_train[0:num]  # (n,)


# for gpflow function data and test_data
x_train = x_train.reshape(num, -1).astype(np.float64)  # (n, 28*28)
y_train = y_train.reshape(num, -1).astype(np.float64)  # (n,1)
x_test = x_test.reshape(NUM_TEST_DATA, -1).astype(np.float64)
t_test = y_test.reshape(NUM_TEST_DATA, -1).astype(np.float64)
data = (x_train, y_train)
test_data = (x_test, y_test)

base_k = gpflow.kernels.SquaredExponential()
# conv kernel by sum all the patches
conv_k = gpflow.kernels.Convolutional(base_k, IMAGE_SHAPE, patch_shape)
# base_k.lengthscales = gpflow.Parameter(1.0, transform=positive_with_min())
# # Weight scale and variance are non-identifiable. We also need to prevent variance from shooting off crazily.
# conv_k.base_kernel.variance = gpflow.Parameter(1.0, transform=constrained())
# conv_k.weights = gpflow.Parameter(conv_k.weights.numpy(), transform=max_abs_1())
patches = conv_k.get_patches(x_train)
patches_np = patches.numpy()
patches_np1 = patches_np.reshape(-1, 25)
patches_np2 = patches_np1.reshape(576, 5, 5)
patches_uniq = np.unique(patches_np)
# conv_f = gpflow.inducing_variables.InducingPatches(
#     np.unique(conv_k.get_patches(x_train).numpy().reshape(-1, 25), axis=0)
# )

for i in range(0, 4):
    plt.subplot(2, 2, i + 1)
    plt.imshow(patches_np2[i + 130, :, :], cmap="gray_r")
plt.show()
