# Gaussian processes and neural networks

Despite the success of deep learning in many application areas, neural networks lack of predictive uncertainty estimates. Gaussian processes, as a Bayesian non-parametric model provide the uncertainty quantification and full mathematical interpretation.

## Methodology
Multiclass classification on MNIST dataset, w/o convolutional struture <br/>
Sparse Gaussian process to reduce the complexity<br/>
Variantional inference<br/>
Optimization with Adam(1st derivate based) and L-BFGS-B(2nd derivative based) methods<br/>

### Environment
tensorflow == 2.3
tensorflow_probability == 0.11.1
python == 3.8
gpflow == 2.1.4
