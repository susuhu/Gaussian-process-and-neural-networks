# using gpflow convolutional kernel

import time
import numpy as np
import matplotlib.pyplot as plt
import gpflow
from gpflow.utilities import print_summary, set_trainable

import tensorflow as tf
import tensorflow_probability as tfp


# Load and prepare the MNIST dataset. Convert the samples from integers to floating-point numbers:
mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# # plot images
# for i in range(10):
#     plt.subplot(2, 5, i + 1)
#     plt.imshow(x_train[i], cmap="gray_r")
#     plt.title("label: %d" % y_train[i])
# plt.show()


# set numbers
NUM_TRAIN_DATA = x_train.shape[0]
NUM_TEST_DATA = x_test.shape[0]
MAXITER = 100
H = W = 28  # width and height
IMAGE_SHAPE = [H, W]

# # small batch data for test run
# x_train = x_train[0:NUM_TRAIN_DATA]  # (n,28,28)
# y_train = y_train[0:NUM_TRAIN_DATA]  # (n,)
# x_test = x_test[NUM_TRAIN_DATA:NUM_TEST_DATA]
# y_test = y_test[NUM_TRAIN_DATA:NUM_TEST_DATA]

# for gpflow data and test_data
x_train = x_train.reshape(NUM_TRAIN_DATA, -1).astype(np.float64)  # (n, 28*28)
y_train = y_train.reshape(NUM_TRAIN_DATA, -1).astype(np.float64)  # (n,1)
x_test = x_test.reshape(NUM_TEST_DATA, -1).astype(np.float64)
t_test = y_test.reshape(NUM_TEST_DATA, -1).astype(np.float64)

data = (x_train, y_train)
test_data = (x_test, y_test)


# GPflow
f64 = lambda x: np.array(x, dtype=np.float64)
max_abs_1 = lambda: tfp.bijectors.AffineScalar(shift=f64(-2.0), scale=f64(4.0))(
    tfp.bijectors.Sigmoid()
)

patch_shape = [3, 3]
base_k = gpflow.kernels.SquaredExponential(variance=1., lengthscales=1.)
conv_k = gpflow.kernels.Convolutional(base_k, IMAGE_SHAPE, patch_shape= [5, 5])
conv_k.weights = gpflow.Parameter(conv_k.weights.numpy(), transform=max_abs_1())
conv_f = gpflow.inducing_variables.InducingPatches(
    np.unique(conv_k.get_patches(x_train).numpy().reshape(-1, 25), axis=0)
)

conv_m = gpflow.models.SVGP(conv_k, gpflow.likelihoods.Bernoulli(), conv_f)

set_trainable(conv_m.inducing_variable, False)
set_trainable(conv_m.kernel.base_kernel.variance, False)
set_trainable(conv_m.kernel.weights, False)

conv_training_loss_closure = conv_m.training_loss_closure(data, compile=True)
conv_elbo = lambda: -conv_training_loss_closure().numpy()
print("conv elbo before training: %.4e" % conv_elbo())

start_time = time.time()
res = gpflow.optimizers.Scipy().minimize(
    conv_training_loss_closure,
    variables=conv_m.trainable_variables,
    method="l-bfgs-b",
    options={"disp": True, "maxiter": MAXITER / 10},
)
print(f"{res.nfev / (time.time() - start_time):.3f} iter/s")

set_trainable(conv_m.kernel.base_kernel.variance, True)
res = gpflow.optimizers.Scipy().minimize(
    conv_training_loss_closure,
    variables=conv_m.trainable_variables,
    method="l-bfgs-b",
    options={"disp": True, "maxiter": MAXITER},
)
train_acc = np.mean((conv_m.predict_y(x_train)[0] > 0.5).numpy().astype("float") == y_train)
test_acc = np.mean((conv_m.predict_y(x_test)[0] > 0.5).numpy().astype("float") == y_test)
print(f"Train acc: {train_acc * 100}%\nTest acc : {test_acc*100}%")
print("conv elbo after training: %.4e" % conv_elbo())

res = gpflow.optimizers.Scipy().minimize(
    conv_training_loss_closure,
    variables=conv_m.trainable_variables,
    method="l-bfgs-b",
    options={"disp": True, "maxiter": MAXITER},
)
train_acc = np.mean((conv_m.predict_y(x_train)[0] > 0.5).numpy().astype("float") == y_train)
test_acc = np.mean((conv_m.predict_y(x_test)[0] > 0.5).numpy().astype("float") == y_test)
print(f"Train acc: {train_acc * 100}%\nTest acc : {test_acc*100}%")
print("conv elbo after training: %.4e" % conv_elbo())

set_trainable(conv_m.kernel.weights, True)
res = gpflow.optimizers.Scipy().minimize(
    conv_training_loss_closure,
    variables=conv_m.trainable_variables,
    method="l-bfgs-b",
    options={"disp": True, "maxiter": MAXITER},
)
train_acc = np.mean((conv_m.predict_y(x_train)[0] > 0.5).numpy().astype("float") == y_train)
test_acc = np.mean((conv_m.predict_y(x_test)[0] > 0.5).numpy().astype("float") == y_train)
print(f"Train acc: {train_acc * 100}%\nTest acc : {test_acc*100}%")
print("conv elbo after training: %.4e" % conv_elbo())

gpflow.utilities.print_summary(conv_m)