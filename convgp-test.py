# using gpflow convolutional kernel

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

# # plot images
# for i in range(10):
#     plt.subplot(2, 5, i + 1)
#     plt.imshow(x_train[i], cmap="gray_r")
#     plt.title("label: %d" % y_train[i])
# plt.show()


# set numbers
NUM_TRAIN_DATA = np.array([60, 600, 6000, 60000])
NUM_TEST_DATA = x_test.shape[0]
MAXITER = 100
H = W = 28  # width and height
IMAGE_SHAPE = [H, W]

# different training data size run
for num in NUM_TRAIN_DATA:
    x_train = x_train[0:num]  # (n,28,28)
    y_train = y_train[0:num]  # (n,)

    # for gpflow function data and test_data
    x_train = x_train.reshape(num, -1).astype(np.float64)  # (n, 28*28)
    y_train = y_train.reshape(num, -1).astype(np.float64)  # (n,1)
    x_test = x_test.reshape(NUM_TEST_DATA, -1).astype(np.float64)
    t_test = y_test.reshape(NUM_TEST_DATA, -1).astype(np.float64)
    data = (x_train, y_train)
    test_data = (x_test, y_test)

    # GPflow
    f64 = lambda x: np.array(x, dtype=np.float64)
    max_abs_1 = lambda: tfp.bijectors.AffineScalar(shift=f64(-2.0), scale=f64(4.0))(
        tfp.bijectors.Sigmoid()
    )

    patch_shape = [5, 5]
    # base kernel, setting variance=1 and lengthschales=1 as a starting point
    base_k = gpflow.kernels.SquaredExponential(variance=1., lengthscales=1.)
    # conv kernel by sum all the patches
    conv_k = gpflow.kernels.Convolutional(base_k, IMAGE_SHAPE, patch_shape)
    conv_k.weights = gpflow.Parameter(conv_k.weights.numpy(), transform=max_abs_1())
    conv_f = gpflow.inducing_variables.InducingPatches(
        np.unique(conv_k.get_patches(x_train).numpy().reshape(-1, 25), axis=0)
    )

    # sparse variance Gaussian Process
    conv_m = gpflow.models.SVGP(conv_k, gpflow.likelihoods.Bernoulli(), conv_f)

    conv_training_loss_closure = conv_m.training_loss_closure(data, compile=True)
    conv_elbo = lambda: -conv_training_loss_closure().numpy()
    print("conv elbo before training: %.4e" % conv_elbo())
    print("conv model summary before training:")
    gpflow.utilities.print_summary(conv_m)

    print("TRAINING DATA SIZE IS %d" % num)

    # set variance, weight as trainable parameters
    set_trainable(conv_m.kernel.base_kernel.variance, True)
    set_trainable(conv_m.kernel.weights, True)
    res = gpflow.optimizers.Scipy().minimize(
        conv_training_loss_closure,
        variables=conv_m.trainable_variables,
        method="l-bfgs-b",
        options={"disp": True, "maxiter": MAXITER},
    )
    # results of training variance
    train_acc = np.mean((conv_m.predict_y(x_train)[0] > 0.5).numpy().astype("float") == y_train)
    test_acc = np.mean((conv_m.predict_y(x_test)[0] > 0.5).numpy().astype("float") == y_test)
    print(f"Train acc: {train_acc * 100}%\nTest acc : {test_acc * 100}%")
    print("conv elbo after training: %.4e" % conv_elbo())

    print("conv model summary after training:")
    gpflow.utilities.print_summary(conv_m)
