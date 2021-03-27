
# %%
import numpy as np
import scipy
from scipy import spatial
import matplotlib.pyplot as plt
# %%
def func(a,b,xa,xb):
    f = a**2 * np.exp(-scipy.spatial.distance.cdist(xa,xb,'sqeuclidean')/(b**2))
    return f

# %%
n = 300       # number of points.
m = 10        # number of functions to draw.

# %%
# Sample from the Gaussian process distribution
nb_of_samples = 100  # Number of points in each function
number_of_functions = 5  # Number of functions to sample
# Independent variable samples
X = np.expand_dims(np.linspace(-4, 4, nb_of_samples), 1)
ff1 = func(1,1,X, X)  # Kernel of data points
ff2 = func(1,0.5,X, X)
ff3 = func(0.5,1,X, X)
ff4 = func(0.5,0.5,X, X)
# %%
# Draw samples from the prior at our data points.
# Assume a mean of 0 for simplicity
ys1 = np.random.multivariate_normal(
    mean=np.zeros(nb_of_samples), cov=ff1,
    size=number_of_functions)

ys2 = np.random.multivariate_normal(
    mean=np.zeros(nb_of_samples), cov=ff2,
    size=number_of_functions)


ys3 = np.random.multivariate_normal(
    mean=np.zeros(nb_of_samples), cov=ff3,
    size=number_of_functions)

ys4 = np.random.multivariate_normal(
    mean=np.zeros(nb_of_samples), cov=ff4,
    size=number_of_functions)
# %%
# Plot the sampled functions
plt.figure(figsize=(6, 4), dpi=100)

# plt.suptitle('Squared exponentiated kernel with differnet parameters',fontsize=12)
plt.subplots_adjust(hspace=0.5)
plt.subplot(2,2,1)
plt.xlim([-4, 4])
plt.ylim([-3,3])
for i in range(number_of_functions):
    plt.plot(X, ys1[i], linestyle='-',color ="darkblue")
    plt.title("σ =1.0, l =1.0",fontsize=10)


plt.subplot(2,2,2)
plt.xlim([-4, 4])
plt.ylim([-3,3])
for i in range(number_of_functions):
    plt.plot(X, ys2[i], linestyle='-',color ="darkred")
    plt.title("σ =1.0, l =0.5",fontsize=10)

plt.subplot(2,2,3)
plt.xlim([-4, 4])
plt.ylim([-3,3])
for i in range(number_of_functions):
    plt.plot(X, ys3[i], linestyle='-',color='darkgreen')
    plt.title("σ =0.5, l =1.0",fontsize=10)

plt.subplot(2,2,4)
plt.xlim([-4, 4])
plt.ylim([-3,3])
for i in range(number_of_functions):
    plt.plot(X, ys4[i], linestyle='-',color='orange')
    plt.title("σ =0.5, l =0.5",fontsize=10)

plt.savefig('rbf_kernel_vis.png',bbox_inches='tight')
plt.show()

# %%
