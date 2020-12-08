import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import mnist
import tensorflow_probability as tfp

import gpflow
from gpflow import set_trainable
from gpflow.ci_utils import is_continuous_integration

# ???
from gpflow.models import SVGP

gpflow.config.set_default_float(np.float64)
gpflow.config.set_default_jitter(1e-4)
# gpflow.config.set_default_summary_fmt("test-title")

# for reproducibility of this notebook:
np.random.seed(123)
tf.random.set_seed(42)

# read data
x_train = mnist.train_images()
y_train = mnist.train_labels()
x_test = mnist.test_images()
y_test = mnist.test_labels()

# small batch data for test run
x_train = x_train[0:200]
y_train = y_train[0:200]
x_test = x_test[0:20]
y_test = y_test[0:20]

# plot images
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_train[i], cmap="gray_r")
    plt.title("label: %d" % y_train[i])
plt.show()

# Squared Exponential kernel gpflow
rbf_m: SVGP = gpflow.models.SVGP(
    gpflow.kernels.SquaredExponential(),
    gpflow.likelihoods.Bernoulli(),
    gpflow.inducing_variables.InducingPoints(x_train.copy()),
)
rbf_training_loss_closure = rbf_m.training_loss_closure(x_train, compile=True)
rbf_elbo = lambda: -rbf_training_loss_closure().numpy()
print("RBF elbo before training: %.4e" % rbf_elbo())
