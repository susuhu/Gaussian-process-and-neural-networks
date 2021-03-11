# using gpflow convolutional kernel

import time
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

# import matplotlib.pyplot as plt
import gpflow
from gpflow.utilities import set_trainable, print_summary


# plain GP model
def rbf_model(base_k, x_train, y_train, x_test, y_test, MAXITER, n):
    # data
    data = (x_train, y_train)

    # inducing points
    idx = np.random.choice(x_train.shape[0], n, replace=False)
    induce_x = x_train[idx]

    # plain squared exponential kernel, no convolution
    rbf_m1 = gpflow.models.SVGP(
        base_k,
        gpflow.likelihoods.MultiClass(10),
        gpflow.inducing_variables.InducingPoints(induce_x.copy()),
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

    print_summary(rbf_m1)

    return rbf_m1


# convolutional GP model
def conv_model(conv_k, conv_f, x_train, y_train, x_test, y_test, MAXITER):
    # data
    data = (x_train, y_train)

    # convolutional model
    conv_m1 = gpflow.models.SVGP(conv_k, gpflow.likelihoods.MultiClass(10), conv_f, num_latent_gps=10)

    conv_training_loss_closure = conv_m1.training_loss_closure(data, compile=True)

    # set variance, lengthscale, weight as trainable parameters
    set_trainable(conv_m1.inducing_variable, True)
    set_trainable(conv_m1.kernel.base_kernel.variance, True)
    set_trainable(conv_m1.kernel.base_kernel.lengthscales, True)
    set_trainable(conv_m1.kernel.weights, False)

    # run scipy optimization
    start_time = time.time()
    res = gpflow.optimizers.Scipy().minimize(
        conv_training_loss_closure,
        variables=conv_m1.trainable_variables,
        method="l-bfgs-b",
        options={"disp": True, "maxiter": MAXITER},
    )
    print(f"{res.nfev / (time.time() - start_time):.3f} iter/s")

    print_summary(conv_m1)

    return conv_m1


def my_inducing_points(x, n, base_k, IMAGE_SHAPE, PATCH_SHAPE):
    conv_k = gpflow.kernels.Convolutional(base_k, IMAGE_SHAPE, PATCH_SHAPE)
    patches = np.unique(conv_k.get_patches(x).numpy().reshape(-1, 25), axis=0)
    idx = np.random.choice(patches.shape[0], n, replace=False)
    Z = patches[idx, :]
    return Z


def pred(model, x_train, y_train, x_test, y_test):
    predicted_train_mean, _ = model.predict_y(x_train)
    predicted_test_mean, _ = model.predict_y(x_test)
    # predicted_train = np.argmax(predicted_train_mean.numpy(), axis=1).reshape(
    #     y_train.shape[0], 1
    # )
    train_acc = np.mean((np.argmax(predicted_train_mean.numpy(), axis=1).reshape(y_train.shape[0], 1)) == y_train)
    test_acc = np.mean((np.argmax(predicted_test_mean.numpy(), axis=1).reshape(y_test.shape[0], 1)) == y_test)
    return train_acc, test_acc


def main():
    # Load MNIST dataset
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()  # 60k train, 10k test
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # set numbers
    NUM_TRAIN_DATA = x_train.shape[0]
    NUM_TEST_DATA = x_test.shape[0]
    # NUM_TRAIN_DATA = 20
    # NUM_TEST_DATA = 10
    MAXITER = 100
    IMAGE_SHAPE = [28, 28]
    PATCH_SHAPE = [5, 5]
    M = 500  # size of inducing points

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
    # conv kernel by summing all the patches
    conv_k = gpflow.kernels.Convolutional(base_k, IMAGE_SHAPE, PATCH_SHAPE)
    # apply constraints
    conv_k.base_kernel.lengthscales = gpflow.Parameter(1.0, transform=positive_with_min())
    conv_k.base_kernel.variance = gpflow.Parameter(1.0, transform=constrained())
    conv_k.weights = gpflow.Parameter(conv_k.weights.numpy(), transform=max_abs_1())

    # indcuing points for conv model
    Z = my_inducing_points(x_train, M, base_k, IMAGE_SHAPE, PATCH_SHAPE)
    conv_f = gpflow.inducing_variables.InducingPatches(Z)

    # run model optimization
    rbf_m = rbf_model(base_k, x_train, y_train, x_test, y_test, MAXITER, M)
    conv_m = conv_model(conv_k, conv_f, x_train, y_train, x_test, y_test, MAXITER)

    # run prediction
    rbf_train_acc, rbf_test_acc = pred(rbf_m, x_train, y_train, x_test, y_test)
    conv_train_acc, conv_test_acc = pred(conv_m, x_train, y_train, x_test, y_test)

    print("RBF training accuracy is: %.2f " % rbf_train_acc)
    print("RBF testing accuracy is: %.2f " % rbf_test_acc)
    print("Conv model training accuracy is: %.2f" % conv_train_acc)
    print("Conv model testing accuracy is: %.2f" % conv_test_acc)


if __name__ == "__main__":
    main()
