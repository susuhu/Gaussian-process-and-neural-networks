# using gpflow convolutional kernel

import time
import numpy as np
# import matplotlib.pyplot as plt
import gpflow
from gpflow.utilities import set_trainable

import tensorflow as tf
import tensorflow_probability as tfp


# plain GP model
def rbf_model(base_k, x_train, y_train, x_test, y_test, MAXITER):
    # data
    data = (x_train, y_train)

    # plain squared exponential kernel, no convolution
    rbf_m1 = gpflow.models.SVGP(
        base_k,
        gpflow.likelihoods.MultiClass(10),
        gpflow.inducing_variables.InducingPoints(x_train.copy()),
        num_latent_gps=10,
    )

    rbf_training_loss_closure = rbf_m1.training_loss_closure(data, compile=True)

    # train plain squared exponetial model
    set_trainable(rbf_m1.inducing_variable, True)
    start_time = time.time()
    res = gpflow.optimizers.Scipy().minimize(
        rbf_training_loss_closure,
        variables=rbf_m1.trainable_variables,
        method="l-bfgs-b",
        options={"disp": True, "maxiter": MAXITER},
    )
    print(f"{res.nfev / (time.time() - start_time):.3f} iter/s")

    rbf_train_acc = np.mean(
        (np.argmax(rbf_m1.predict_y(x_train)[0].numpy(), axis=1).reshape(y_train.shape[0], 1)) == y_train
    )
    rbf_test_acc = np.mean(
        (np.argmax(rbf_m1.predict_y(x_test)[0].numpy(), axis=1).reshape(y_test.shape[0], 1)) == y_test
    )
    return rbf_train_acc, rbf_test_acc


# convolutional GP model
def conv_model(base_k, conv_k, conv_f, x_train, y_train, x_test, y_test, MAXITER):
    # data
    data = (x_train, y_train)

    # convolutional model
    conv_m = gpflow.models.SVGP(conv_k, gpflow.likelihoods.MultiClass(10), conv_f, num_latent_gps=10)

    conv_training_loss_closure = conv_m.training_loss_closure(data, compile=True)

    # set variance, lengthscale, weight as trainable parameters
    set_trainable(conv_m.inducing_variable, True)
    set_trainable(conv_m.kernel.base_kernel.variance, True) 
    set_trainable(conv_m.kernel.base_kernel.lengthscales, True)
    set_trainable(conv_m.kernel.weights, True) 

    # run optimization
    res = gpflow.optimizers.Scipy().minimize(
        conv_training_loss_closure,
        variables=conv_m.trainable_variables,
        method="l-bfgs-b",
        options={"disp": True, "maxiter": MAXITER},
    )

    # results of training variance
    conv_train_acc = np.mean(
        (np.argmax(conv_m.predict_y(x_train)[0].numpy(), axis=1).reshape(y_train.shape[0], 1)) == y_train
    )
    conv_test_acc = np.mean(
        (np.argmax(conv_m.predict_y(x_test)[0].numpy(), axis=1).reshape(y_test.shape[0], 1)) == y_test
    )

    print(np.argmax(conv_m.predict_y(x_train)[0].numpy(), axis=1))
    print(y_train)

    return conv_train_acc, conv_test_acc


def main():
    # Load MNIST dataset
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()  # 60k train, 10k test
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # set numbers
    NUM_TRAIN_DATA = x_train.shape[0]
    NUM_TEST_DATA = x_test.shape[0]
    # NUM_TRAIN_DATA = 8
    # NUM_TEST_DATA = 5
    MAXITER = 5
    H = W = 28  # width and height
    IMAGE_SHAPE = [H, W]
    PATCH_SHAPE = [5, 5]


    # # for small batch test
    # x_train = x_train[0:NUM_TRAIN_DATA]  # (n,28,28)
    # y_train = y_train[0:NUM_TRAIN_DATA]  # (n,)
    # x_test = x_test[0:NUM_TEST_DATA]
    # y_test = y_test[0:NUM_TEST_DATA]

    # process data for gpflow
    x_train = x_train.reshape(NUM_TRAIN_DATA, -1).astype(np.float64)  # (n, 28*28)
    y_train = y_train.reshape(NUM_TRAIN_DATA, -1).astype(np.float64)  # (n,1)
    x_test = x_test.reshape(NUM_TEST_DATA, -1).astype(np.float64)
    y_test = y_test.reshape(NUM_TEST_DATA, -1).astype(np.float64)

    # set constraints
    f64 = lambda x: np.array(x, dtype=np.float64)
    positive_with_min = lambda: tfp.bijectors.AffineScalar(shift=f64(1e-4))(tfp.bijectors.Softplus())
    constrained = lambda: tfp.bijectors.AffineScalar(shift=f64(1e-4), scale=f64(100.0))(tfp.bijectors.Sigmoid())
    max_abs_1 = lambda: tfp.bijectors.AffineScalar(shift=f64(-2.0), scale=f64(4.0))(tfp.bijectors.Sigmoid())

    # base kernel
    base_k = gpflow.kernels.SquaredExponential()
    # conv kernel by sum all the patches
    conv_k = gpflow.kernels.Convolutional(base_k, IMAGE_SHAPE, PATCH_SHAPE)
    # apply constraints
    conv_k.base_kernel.lengthscales = gpflow.Parameter(1.0, transform=positive_with_min())
    conv_k.base_kernel.variance = gpflow.Parameter(1.0, transform=constrained())
    conv_k.weights = gpflow.Parameter(conv_k.weights.numpy(), transform=max_abs_1())
    conv_f = gpflow.inducing_variables.InducingPatches(
        np.unique(conv_k.get_patches(x_train).numpy().reshape(-1, 25), axis=0)
    )

    rbf_train_acc, rbf_test_acc = rbf_model(base_k, x_train, y_train, x_test, y_test, MAXITER)
    conv_train_acc, conv_test_acc = conv_model(base_k, conv_k, conv_f, x_train, y_train, x_test, y_test, MAXITER)

    print("rbf training and testing accuracy:")
    print(rbf_train_acc)
    print(rbf_test_acc)
    print("convolutional GP training and testing accuracy:")
    print(conv_train_acc)
    print(conv_test_acc)


if __name__ == "__main__":
    main()
