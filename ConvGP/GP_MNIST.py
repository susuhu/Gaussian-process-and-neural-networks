"""
Guassian processes on MNIST image classification problem
using plain RBF model and convolutional kernel
"""
import time
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import gpflow
from gpflow.utilities import set_trainable, print_summary
import matplotlib.pyplot as plt


# GP model without convolutional structure
def rbf_model(base_k, data, MAXITER, n, num_class):
    # inducing points
    idx = np.random.choice(data[0].shape[0], n, replace=False)
    induce_x = data[0][idx]

    # plain squared exponential kernel, no convolution
    rbf_m1 = gpflow.models.SVGP(
        base_k,
        gpflow.likelihoods.MultiClass(num_class),
        gpflow.inducing_variables.InducingPoints(induce_x.copy()),
        num_latent_gps=num_class,
    )

    # set traininable
    set_trainable(rbf_m1.kernel.variance, True)
    set_trainable(rbf_m1.kernel.lengthscales, True)
    set_trainable(rbf_m1.inducing_variable, True)

    return rbf_m1


# GP model with convolutional structure
def conv_model(conv_k, conv_f, data, num_class):
    # convolutional model
    conv_m1 = gpflow.models.SVGP(
        conv_k,
        gpflow.likelihoods.MultiClass(num_class),
        conv_f,
        num_latent_gps=num_class,
    )

    # set variance, lengthscale, weight as trainable parameters
    set_trainable(conv_m1.inducing_variable, True)
    set_trainable(conv_m1.kernel.base_kernel.variance, True)
    set_trainable(conv_m1.kernel.base_kernel.lengthscales, True)
    set_trainable(conv_m1.kernel.weights, False)

    return conv_m1


# L-BFGS-B optimizer
def run_lbfgs(model, MAXITER, data, M):
    training_loss = model.training_loss_closure(data, compile=True)

    start_time = time.time()
    opt = gpflow.optimizers.Scipy().minimize(
        training_loss,
        variables=model.trainable_variables,
        method="l-bfgs-b",
        options={"disp": True},
    )

    print(f"{(time.time() - start_time)/MAXITER:.3f} s/iter")
    return opt


# Adam optimizer
def run_adam(model, MAXITER, data):
    """
    Utility function running the Adam optimizer
    """

    # Create an Adam Optimizer action
    losses = []
    training_loss = model.training_loss_closure(data, compile=True)
    optimizer = tf.optimizers.Adam()

    @tf.function
    def optimization_step():
        optimizer.minimize(training_loss, model.trainable_variables)

    start_time = time.time()
    for step in range(MAXITER):
        start_time = time.time()
        optimization_step()

        losses.append(training_loss().numpy())
    print(f"{(time.time() - start_time)/MAXITER:.3f} s/iter")

    return losses


# Initialize inducing points/patches
def my_inducing_points(x, n, base_k, IMAGE_SHAPE, PATCH_SHAPE):
    conv_k = gpflow.kernels.Convolutional(base_k, IMAGE_SHAPE, PATCH_SHAPE)
    patches = np.unique(conv_k.get_patches(x).numpy().reshape(-1, 81), axis=0)
    print("patch shape: {0}".format(patches.shape))
    idx = np.random.choice(patches.shape[0], n, replace=False)
    Z = patches[idx, :]
    print("Z shape: {0}".format(Z.shape))
    return Z


# Prediction
def pred(model, x_train, y_train, x_test, y_test):
    predicted_train_mean, _ = model.predict_y(x_train)
    predicted_test_mean, _ = model.predict_y(x_test)
    train_acc = np.mean(
        (np.argmax(predicted_train_mean.numpy(), axis=1).reshape(y_train.shape[0], 1))
        == y_train
    )
    test_acc = np.mean(
        (np.argmax(predicted_test_mean.numpy(), axis=1).reshape(y_test.shape[0], 1))
        == y_test
    )
    return train_acc, test_acc


# Plot inducing patches
def plot_patches(pat, n, PATCH_SHAPE):
    plt.figure(figsize=(5, 5))
    for i in range(n):
        plt.subplot(2, n / 2, i + 1)
        plt.imshow(pat[i, :].numpy().reshape(*PATCH_SHAPE))
        plt.axis("off")
        plt.savefig("Results/testpatches.png", bbox_inches="tight")


def main():
    # Load MNIST dataset
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()  # 60k train, 10k test
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # set numbers
    NUM_TRAIN_DATA = x_train.shape[0]
    NUM_TEST_DATA = x_test.shape[0]
    MAXITER = 100
    IMAGE_SHAPE = [28, 28]
    PATCH_SHAPE = [9, 9]
    M = 100
    num_class = 10

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

    data = (x_train, y_train)

    # set constraints
    f64 = lambda x: np.array(x, dtype=np.float64)
    positive_with_min = lambda: tfp.bijectors.AffineScalar(shift=f64(1e-4))(
        tfp.bijectors.Softplus()
    )
    constrained = lambda: tfp.bijectors.AffineScalar(shift=f64(1e-4), scale=f64(100.0))(
        tfp.bijectors.Sigmoid()
    )
    max_abs_1 = lambda: tfp.bijectors.AffineScalar(shift=f64(-2.0), scale=f64(4.0))(
        tfp.bijectors.Sigmoid()
    )

    # base kernel
    base_k = gpflow.kernels.SquaredExponential()
    # conv kernel by summing all the patches
    conv_k = gpflow.kernels.Convolutional(base_k, IMAGE_SHAPE, PATCH_SHAPE)
    # apply constraints to trainable varialbes
    conv_k.base_kernel.lengthscales = gpflow.Parameter(
        1.0, transform=positive_with_min()
    )
    conv_k.base_kernel.variance = gpflow.Parameter(1.0, transform=constrained())
    conv_k.weights = gpflow.Parameter(conv_k.weights.numpy(), transform=max_abs_1())

    # initialize indcuing points for conv model
    Z = my_inducing_points(x_train, M, base_k, IMAGE_SHAPE, PATCH_SHAPE)
    conv_f = gpflow.inducing_variables.InducingPatches(Z)

    # models
    rbf_m = rbf_model(base_k, data, MAXITER, M, num_class)
    conv_m = conv_model(conv_k, conv_f, data, num_class)

    # run l-bfgs-b optimization
    rbf_lbfgs = run_lbfgs(rbf_m, MAXITER, data)
    conv_lbfgs = run_lbfgs(conv_m, MAXITER, data)

    # #run adam
    # loss_rbf_adam = run_adam(rbf_m, MAXITER, data)
    # loss_conv_adam = run_adam(conv_m, MAXITER, data)
    # print(loss_conv_adam)

    # model summary
    print_summary(rbf_m)
    print_summary(conv_m)

    # run prediction
    rbf_train_acc, rbf_test_acc = pred(rbf_m, x_train, y_train, x_test, y_test)
    conv_train_acc, conv_test_acc = pred(conv_m, x_train, y_train, x_test, y_test)

    print("RBF training accuracy is: %.2f " % rbf_train_acc)
    print("RBF testing accuracy is: %.2f " % rbf_test_acc)
    print("Conv model training accuracy is: %.2f" % conv_train_acc)
    print("Conv model testing accuracy is: %.2f" % conv_test_acc)

    # # get and plot optimized the inducing points
    # opt_Z = conv_m.inducing_variable.Z
    # plot_patches(opt_Z, M, PATCH_SHAPE)


if __name__ == "__main__":
    main()
