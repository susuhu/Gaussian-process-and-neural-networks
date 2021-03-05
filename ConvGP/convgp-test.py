# using gpflow convolutional kernel

import time
import numpy as np
import matplotlib.pyplot as plt
import gpflow
from gpflow.utilities import set_trainable

import tensorflow as tf
import tensorflow_probability as tfp

# Load and prepare the MNIST dataset
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()  # 60k train, 10k test
x_train, x_test = x_train / 255.0, x_test / 255.0


# set numbers
# NUM_TRAIN_DATA = np.array([60, 600, 6000, 60000])  # separate training sample size
NUM_TRAIN_DATA = 8
# NUM_TEST_DATA = x_test.shape[0]
NUM_TEST_DATA = 8
MAXITER = 3
H = W = 28  # width and height
IMAGE_SHAPE = [H, W]
patch_shape = [5, 5]


# different training data size run

x_train = x_train[0:NUM_TRAIN_DATA]  # (n,28,28)
y_train = y_train[0:NUM_TRAIN_DATA]  # (n,)
x_test = x_test[0:NUM_TEST_DATA]
y_test = y_test[0:NUM_TEST_DATA]

# for gpflow function data and test_data
x_train = x_train.reshape(NUM_TRAIN_DATA, -1).astype(np.float64)  # (n, 28*28)
y_train = y_train.reshape(NUM_TRAIN_DATA, -1).astype(np.float64)  # (n,1)
x_test = x_test.reshape(NUM_TEST_DATA, -1).astype(np.float64)
y_test = y_test.reshape(NUM_TEST_DATA, -1).astype(np.float64)
data = (x_train, y_train)
test_data = (x_test, y_test)

# set constraint
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
# conv kernel by sum all the patches
conv_k = gpflow.kernels.Convolutional(base_k, IMAGE_SHAPE, patch_shape)
conv_k.base_kernel.lengthscales = gpflow.Parameter(
    1.0, transform=positive_with_min()
)
conv_k.base_kernel.variance = gpflow.Parameter(1.0, transform=constrained())
conv_k.weights = gpflow.Parameter(conv_k.weights.numpy(), transform=max_abs_1())
conv_f = gpflow.inducing_variables.InducingPatches(
    np.unique(conv_k.get_patches(x_train).numpy().reshape(-1, 25), axis=0)
)


# plain squared exponential kernel, no convolution
rbf_m1 = gpflow.models.SVGP(
    gpflow.kernels.SquaredExponential(),
    gpflow.likelihoods.MultiClass(10),
    gpflow.inducing_variables.InducingPoints(x_train.copy()),
    num_latent_gps=10,
)

rbf_training_loss_closure = rbf_m1.training_loss_closure(data, compile=True)
# rbf_elbo = lambda: -rbf_training_loss_closure().numpy()
# print("RBF elbo before training: %.4e" % rbf_elbo())
# rbf_elbo_a = [] 
# rbf_elbo_a = np.append(rbf_elbo_a, rbf_elbo())

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

rbf_train_acc_list = []
rbf_test_acc_list = []  
rbf_elbo_b = []  

rbf_train_acc = np.mean(
    (np.argmax(rbf_m1.predict_y(x_train)[0].numpy(),axis = 1).reshape(y_train.shape[0],1)) == y_train     
)
rbf_test_acc = np.mean(
    (np.argmax(rbf_m1.predict_y(x_test)[0].numpy(),axis = 1).reshape(y_test.shape[0],1)) == y_test
)

# print(f"Train acc: {rbf_train_acc * 100}%\nTest acc : {rbf_test_acc*100}%")
# print("RBF elbo after training: %.4e" % rbf_elbo())

rbf_train_acc_list = np.append(rbf_train_acc_list, rbf_train_acc)
rbf_test_acc_list = np.append(rbf_test_acc_list, rbf_test_acc)
# rbf_elbo_b = np.append(rbf_elbo_b, rbf_elbo())

# sparse variance Gaussian Process
# num_latent_gps is the number of latent processes to use, defaults to 1
conv_m = gpflow.models.SVGP(conv_k, gpflow.likelihoods.MultiClass(10), conv_f, num_latent_gps=10)

conv_training_loss_closure = conv_m.training_loss_closure(data, compile=True)
# conv_elbo = conv_m.elbo(data)
# print("conv elbo before training: %.4e" % conv_elbo)
# print("conv model summary before training:")
# gpflow.utilities.print_summary(conv_m)

# set variance, lengthscale, weight as trainable parameters
set_trainable(conv_m.inducing_variable, True)
set_trainable(
    conv_m.kernel.base_kernel.variance, True
)  # SquaredExponential variance
set_trainable(
    conv_m.kernel.base_kernel.lengthscales, True
)  # SquaredExponential lengthscales
set_trainable(conv_m.kernel.weights, True)  # conv kernel weights

# run inference
res = gpflow.optimizers.Scipy().minimize(
    conv_training_loss_closure,
    variables=conv_m.trainable_variables,
    method="l-bfgs-b",
    options={"disp": True, "maxiter": MAXITER},
)

# results of training variance
conv_train_acc_list = []  
conv_test_acc_list = [] 
# conv_m.predict_y(x_train)[0] = mean, conv_m.predict_y(x_train)[1] = variance
# conv_train_acc = np.mean(
#     (conv_m.predict_y(x_train)[0] > 0.5).numpy().astype("float") == y_train
# )
conv_train_acc = np.mean(
    (np.argmax(conv_m.predict_y(x_train)[0].numpy(), axis= 1).reshape(y_train.shape[0],1)) == y_train
)
conv_test_acc = np.mean(
    (np.argmax(conv_m.predict_y(x_test)[0].numpy(), axis= 1).reshape(y_test.shape[0],1)) == y_test
)
# print(
#     f"Train acc: {conv_train_acc * 100}%\nTest acc : {conv_test_acc * 100}% with sample size: {TRAINING_NUM}"
# )
# print("conv elbo after training: %.4e" % conv_elbo())


conv_train_acc_list = np.append(conv_train_acc_list, conv_train_acc)
conv_test_acc_list = np.append(conv_test_acc_list, conv_test_acc)

# gpflow.utilities.print_summary(conv_m)


print("rbf training accurary: " )
print(rbf_train_acc_list)
print("rbf test accurary: " )
print(rbf_test_acc_list)
print("conv training accurary: " )
print(conv_train_acc_list)
print("conv test accurary: " )
print(conv_test_acc_list)


