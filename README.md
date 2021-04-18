# Gaussian processes and neural networks

Despite the success of deep learning in many application areas, neural networks lack of predictive uncertainty estimates. Gaussian processes, as a Bayesian non-parametric model provide the uncertainty quantification and full mathematical interpretation. But scabality remains the biggest challenge in Gaussian processes. Due to matrics inversion, the complexity is  ![equation](https://latex.codecogs.com/gif.latex?\fn_cm&space;\mathcal{O}&space;(N^3))

## Methodology
* Multiclass classification on MNIST dataset, w/o convolutional struture <br/>
* Sparse Gaussian process to reduce the complexity<br/>
* Variantional inference (minimizing KL-divergence/maxizing ELBO)<br/>
* Optimization with Adam(1st derivate based) and L-BFGS-B(2nd derivative based) methods<br/>

### Environment
tensorflow == 2.3<br/>
tensorflow_probability == 0.11.1<br/>
python == 3.8<br/>
gpflow == 2.1.4


##### Visualization of Gaussian processes on toy regression problem
<img src="https://github.com/susuhu/Gaussian-process-and-neural-networks/blob/master/Results/GPRegression.png" width=60%>
