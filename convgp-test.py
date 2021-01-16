# using gpflow convolutional kernel aaa

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
NUM_TRAIN_DATA =np.array([60, 600, 6000, 60000])
NUM_TEST_DATA = x_test.shape[0]
MAXITER = 1000
H = W = 28  # width and height
IMAGE_SHAPE = [H, W]
patch_shape = [5, 5]

# different training data size run
for num in NUM_TRAIN_DATA:
    x_train = x_train[0: num]  # (n,28,28)
    y_train = y_train[0: num]  # (n,)

    # for gpflow function data and test_data
    x_train = x_train.reshape(num, -1).astype(np.float64)  # (n, 28*28)
    y_train = y_train.reshape(num, -1).astype(np.float64)  # (n,1)
    x_test = x_test.reshape(NUM_TEST_DATA, -1).astype(np.float64)
    t_test = y_test.reshape(NUM_TEST_DATA, -1).astype(np.float64)
    data = (x_train, y_train)
    test_data = (x_test, y_test)

    # constrains lambda functions
    # f64 = lambda x: np.array(x, dtype=np.float64)
    # positive_with_min = lambda: tfp.bijectors.AffineScalar(shift=f64(1e-4))(tfp.bijectors.Softplus())
    # constrained = lambda: tfp.bijectors.AffineScalar(shift=f64(1e-4), scale=f64(100.0))(
    #     tfp.bijectors.Sigmoid()
    # )
    # max_abs_1 = lambda: tfp.bijectors.AffineScalar(shift=f64(-2.0), scale=f64(4.0))(
    #     tfp.bijectors.Sigmoid()
    # )

    # base kernel
    base_k = gpflow.kernels.SquaredExponential()
    # conv kernel by sum all the patches
    conv_k = gpflow.kernels.Convolutional(base_k, IMAGE_SHAPE, patch_shape)
    # base_k.lengthscales = gpflow.Parameter(1.0, transform=positive_with_min())
    # # Weight scale and variance are non-identifiable. We also need to prevent variance from shooting off crazily.
    # conv_k.base_kernel.variance = gpflow.Parameter(1.0, transform=constrained())
    # conv_k.weights = gpflow.Parameter(conv_k.weights.numpy(), transform=max_abs_1())
    conv_f = gpflow.inducing_variables.InducingPatches(
        np.unique(conv_k.get_patches(x_train).numpy().reshape(-1, 25), axis=0)
    )

    # sparse variance Gaussian Process num_data=minibatch
    conv_m = gpflow.models.SVGP(conv_k, gpflow.likelihoods.MultiClass(10), conv_f, num_latent_gps=10)

    conv_training_loss_closure = conv_m.training_loss_closure(data, compile=True)
    conv_elbo = conv_m.elbo(data)
    print("conv elbo before training: %.4e" % conv_elbo)
    print("conv model summary before training:")
    gpflow.utilities.print_summary(conv_m)

    print("TRAINING DATA SIZE IS %d" % num)

    # set variance, lengthscale, weight as trainable parameters
    set_trainable(conv_m.kernel.base_kernel.variance, True)  # SquaredExponential variance
    set_trainable(conv_m.kernel.base_kernel.lengthscales, True)  # SquaredExponential lengthscales
    set_trainable(conv_m.kernel.weights, True)  # conv kernel weights
    res = gpflow.optimizers.Scipy().minimize(
        conv_training_loss_closure,
        variables=conv_m.trainable_variables,
        method="l-bfgs-b",
        options={"disp": True, "maxiter": MAXITER},
    )

    # results of training variance
    # conv_m.predict_y(x_train)[0] = mean, conv_m.predict_y(x_train)[1] = variance
    train_acc = np.mean((conv_m.predict_y(x_train)[0] > 0.5).numpy().astype("float") == y_train)
    test_acc = np.mean((conv_m.predict_y(x_test)[0] > 0.5).numpy().astype("float") == y_test)
    print(f"Train acc: {train_acc * 100}%\nTest acc : {test_acc * 100}%")
    print("conv elbo after training: %.4e" % conv_elbo())

    print("conv model summary after training:")
    gpflow.utilities.print_summary(conv_m)
