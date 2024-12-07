# README

This repository contains code and experiments related to our research on combining multiple kernels in a Sparse Gaussian Process (GP) framework for correlated web traffic forecasting. Our approach uses a variety of kernels (Squared Exponential, Spectral Mixture, Matérn, Linear, and Sinusoidal) and examines their roles in capturing complex temporal patterns. By optimizing the Evidence Lower Bound (ELBO), we tune kernel weights and hyperparameters to explore their contributions to forecast accuracy and uncertainty quantification.

## Files and Directories

- **exp_inducing_points.ipynb**  
  This notebook focuses on the experiment examining the impact of varying the number of inducing points in the sparse GP model. By altering the number of inducing points, we investigate how the computational trade-offs and representational fidelity affect performance metrics such as MSE and NLPD.

- **exp_kernel_weights.ipynb**  
  In this notebook, we test kernel optimization on all filtered datasets (Soccer, Politics, and Technology). We run experiments to determine how different kernels contribute to the model and whether some kernels dominate or if their weights remain relatively evenly distributed. The aim is to identify the kernels most critical for accurate forecasting.

- **exp_step_size.ipynb**  
  This notebook explores the effect of varying the step size used in ELBO maximization. By experimenting with different step sizes, we analyze how this parameter influences convergence, kernel weight adjustments, predictive accuracy, and the model’s uncertainty estimates.

- **kernels.py**  
  This Python file contains implementations of the various kernels used in the experiments, including the Squared Exponential, Spectral Mixture, Matérn, Linear, and Sinusoidal kernels. Each kernel includes relevant hyperparameters and their functions for computing covariance.

- **data.py**  
  The `data.py` module provides functions for data preprocessing and manipulation. It includes methods for:
  - Splitting input-output matrices.
  - Cleaning and filtering datasets.
  - Applying median filtering or normalization when needed.

- **plot.py**  
  `plot.py` provides code for visualization. It includes functions to generate time series plots, forecast predictions, and diagnostic plots (such as ELBO curves or kernel weight distributions) for better interpretability of results.

- **test_kernels.ipynb**  
  This notebook is a preliminary environment for testing kernels individually. It ensures that kernels are correctly implemented and behave as expected before integrating them into the sparse GP framework.

- **test_simple2D.ipynb**  
  In `test_simple2D.ipynb`, we run a simplified toy experiment on a two-dimensional dataset. This controlled scenario helps us debug the pipeline and verify that our sparse GP setup and kernel combinations work as intended on simpler data before applying them to the more complex web traffic datasets.

- **sparse_gp.py**  
  This file implements the Sparse Gaussian Process model and the ELBO framework. It includes:
  - The variational inference approach for optimizing inducing points and kernel hyperparameters.
  - Functions for computing the ELBO, predictive distributions, and other core components of the sparse GP machinery.

## Reference Paper

The accompanying paper, *"Kernel Combinations for Sparse Gaussian Processes in Correlated Web Traffic Forecasting"*, provides the theoretical background, methodological details, and empirical findings of our work. It covers:

- **Introduction:**  
  Motivation for using Sparse GPs and multiple kernels for forecasting correlated time series.

- **Related Work:**  
  Discussion of kernel methods, sparse approximations for GPs, and the concept of synchronization in time series forecasting.

- **Methods:**  
  Description of the dataset, data preprocessing methods, model formulation, and inference via ELBO optimization.

- **Experiments:**  
  Three main experiments are described:
  1. **Step Size Impact:** How step size selection affects convergence, accuracy, and uncertainty in the sparse GP model.
  2. **Inducing Points Variation:** How changing the number of inducing points influences computational efficiency and predictive quality.
  3. **Kernel Weight Optimization:** How individual kernels contribute to forecasting performance, and whether certain kernels are more critical than others.

- **Discussion:**  
  Analysis of the results, insights into kernel combinations, and recommendations for future work. The paper also highlights challenges such as model overconfidence and suggests directions for improving calibration and interpretability.

## How to Use

1. **Data Preparation:**  
   Ensure you have the Wikipedia Traffic Data Exploration dataset or your chosen time series data. Use `data.py` to clean and process the data into the required format.

2. **Running Experiments:**  
   - For step size experiments, open `exp_step_size.ipynb` and run the cells sequentially.
   - For inducing points experiments, open `exp_inducing_points.ipynb`.
   - For kernel weight optimization, open `exp_kernel_weights.ipynb`.

   Before running these, you may test the kernel implementations in `test_kernels.ipynb` and try the toy dataset in `test_simple2D.ipynb` to confirm the pipeline works.

3. **Model and Plotting Utilities:**  
   - Use `sparse_gp.py` for the GP model setup.
   - Use `plot.py` functions to visualize results.

## Additional Notes

- The optimization problem is non-convex. Different initializations of hyperparameters, step sizes, or noise levels may lead to varied outcomes.
- The code and experiments are structured to encourage exploration. Users can tweak parameters, kernel sets, and data subsets to further investigate the properties of sparse GPs and kernel combinations.

## License

This project is for research and educational purposes. Please cite the accompanying paper if you use any of the methods or code in your work.
