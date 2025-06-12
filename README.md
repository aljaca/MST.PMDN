# Deep Multivariate Skew t-Parsimonious Mixture Density Network (MST-PMDN) 
### Alex J. Cannon [alex.cannon@ec.gc.ca](mailto:alex.cannon@ec.gc.ca?subject=MST-PMDN)

A ['torch for R'](https://torch.mlverse.org/) implementation of a distributional regression model based on a Multivariate Skew t-Parsimonious Mixture Density Network (MST-PMDN). The MST-PMDN framework represents complicated joint output distributions as mixtures of multivariate skew-t components. A volume (L)-shape (A)-orientation (D) (LAD) eigenvalue decomposition parameterization provides a tractable, interpretable, and parsimonious representation of the components' scale matrices, while explicit modeling of skewness and heavy tails can represent asymmetric behavior and tail dependence observed in real-world data. Specifically, parameters of a mixture of multivariate skew t distributions that describe a multivariate output are estimated by training a deep learning model with two multi-modal input branches, one for tabular inputs and the other for (optional) image inputs, both appropriately scaled. The two branches are provided as user-defined torch modules. Outputs from each are concatenated and passed through a dense fusion network, which then leads to the MST-PMDN head. In the absence of both branches, the tabular inputs are fed directly into the dense network. Following the approach used in model-based clustering, scale matrices output from the MST-PMDN head are represented using an LAD eigen-decomposition parameterization. LAD attributes, the nu (or degrees of freedom) parameter (n), and the alpha (or skewness) parameter (s) can be forced to be Variable or Equal between mixture components (plus Identity for A and D). For n and s, values of nu and alpha can also be constrained to emulate a multivariate normal (N) distribution. In the case of n, users can specify fixed (F) values for nu by passing an optional `fixed_nu` vector. If an element of `fixed_nu` is set to `NA`, then the value of nu for this component is learned by the network. Different model types are specified by setting the argument `constraint = "EIINN"`, `"VEVFV"`, etc. where each letter position in the argument corresponds, respectively, to each of the LADns attributes. Furthermore, values of  mu (or means) (m), pi (or mixing coefficients) (x), volume-shape-orientation attributes (LAD), nu (or degrees of freedom) (n), and alpha (or skewness) (s) for the mixtures can be made to be independent of inputs by specifying any combination of `constant_attr = "m"`, `"mx"`, ..., `"LADmxns"`.

## Installation

```r
install.packages("https://github.com/aljaca/MST.PMDN/archive/refs/tags/v0.1.0.tar.gz")
```

## Example

```r
library(MST.PMDN)

device <- ifelse(cuda_is_available(), "cuda", "cpu")
set.seed(1)
torch_manual_seed(1)

# Significant wave height and storm surge data and covariates
x <- wave_surge$x
x_image <- wave_surge$x_image
y <- wave_surge$y

# The TabularModule takes an input vector of length input_dim, runs it
# through two dense layers (input_dim→32 and 32→16) each with
# batch-norm (BN), ReLU and 50 %  dropout, then applies a final 16→16
# linear layer plus ReLU to produce a 16-dimensional output.
tabular_module <- nn_module(
  "TabularModule",
  initialize = function(
    input_dim,
    hidden_dims,
    output_dim,
    dropout_rate
  ) {
    # Number of hidden layers
    if (is.null(hidden_dims) || length(hidden_dims) == 0) {
      # No hidden layers
      self$n_hidden_layers <- 0
      self$hidden_dims <- c()
    } else if (!is.vector(hidden_dims) && !is.list(hidden_dims)) {
      # Single hidden size passed, wrap into vector
      self$hidden_dims <- c(hidden_dims)
      self$n_hidden_layers <- length(self$hidden_dims)
    } else {
      # Vector or list of hidden sizes
      self$hidden_dims <- hidden_dims
      self$n_hidden_layers <- length(self$hidden_dims)
    }
    # Store output size and dropout rate
    self$output_dim <- output_dim
    self$dropout_rate <- dropout_rate
    # Module lists for linear layers, batch-norms, dropouts
    self$layers <- nn_module_list()
    self$bns <- nn_module_list()
    if (self$dropout_rate > 0) {
      self$dropouts <- nn_module_list()
    }
    # Build hidden layers
    current_dim <- input_dim
    if (self$n_hidden_layers > 0) {
      for (i in seq_len(self$n_hidden_layers)) {
        # Linear transform
        self$layers$append(
          nn_linear(current_dim, self$hidden_dims[[i]])
        )
        # Batch normalization on hidden size
        self$bns$append(
          nn_batch_norm1d(self$hidden_dims[[i]])
        )
        # Optional dropout after activation
        if (self$dropout_rate > 0) {
          self$dropouts$append(
            nn_dropout(p = self$dropout_rate)
          )
        }
        # Update input size for next layer
        current_dim <- self$hidden_dims[[i]]
      }
    }
    # Final linear layer: last hidden (or input) → output_dim
    self$layers$append(
      nn_linear(current_dim, output_dim)
    )
  },
  forward = function(x) {
    # Pass through each hidden layer
    for (i in seq_len(self$n_hidden_layers)) {
      x <- self$layers[[i]](x)  # linear
      x <- self$bns[[i]](x)     # batch-norm
      x <- nnf_relu(x)          # activation
      # Apply dropout if configured
      if (self$dropout_rate > 0 && !is.null(self$dropouts[[i]])) {
        x <- self$dropouts[[i]](x)
      }
    }
    # Final projection and activation
    x <- self$layers[[length(self$layers)]](x)
    x <- nnf_relu(x)
    x
  }
)

# The ImageModule accepts a 2×32×32 image, applies a 3×3 conv (2→16)
# with BN, ReLU  and 2×2 max-pool (→16×16), repeats with a 16→32 conv
# + BN, ReLU and max-pool (→8×8), flattens the 32×8×8 tensor to 2048
# units, and then projects it to 32 features via a linear layer, BN,
# and ReLU. Weight penalty (wd_image) is applied during training.
image_module <- nn_module(
  "ImageModule",
  initialize = function(
    in_channels,
    img_size,
    conv_channels,
    kernel_size = 3,
    pool_kernel = 2,
    output_dim = 32
  ) {
    # Store output dim
    self$output_dim <- output_dim
    # Build conv stack
    self$n_conv <- length(conv_channels)
    self$convs <- nn_module_list()
    self$bn_conv <- nn_module_list()
    # Track spatial dim through conv+pool
    spatial <- img_size
    pad <- floor(kernel_size / 2)
    for (i in seq_along(conv_channels)) {
      in_ch <- if (i == 1) in_channels else conv_channels[i-1]
      out_ch <- conv_channels[i]
      # conv keeps spatial size (with padding)
      self$convs$append(
        nn_conv2d(
          in_channels = in_ch,
          out_channels = out_ch,
          kernel_size = kernel_size,
          padding = pad
        )
      )
      self$bn_conv$append(nn_batch_norm2d(out_ch))
      # Pooling halves spatial dims
      spatial <- floor(spatial / pool_kernel)
    }
    # Store pooling layer and computed flatten_dim
    self$pool <- nn_max_pool2d(kernel_size = pool_kernel)
    self$flatten_dim <- tail(conv_channels, 1) * spatial * spatial
    # Final head: linear( flatten_dim → output_dim ) + BN
    self$fc    <- nn_linear(self$flatten_dim, output_dim)
    self$bn_fc <- nn_batch_norm1d(output_dim)
  },
  forward = function(x) {
    # conv → BN → ReLU → pool
    for (i in seq_len(self$n_conv)) {
      x <- self$convs[[i]](x)
      x <- self$bn_conv[[i]](x)
      x <- nnf_relu(x)
      x <- self$pool(x)
    }
    # Flatten and head
    x <- torch_flatten(x, start_dim = 2)
    x <- self$fc(x)
    nnf_relu(self$bn_fc(x))
  }
)

# Instantiate the tabular and image modules
tabular_mod <- tabular_module(
  input_dim = ncol(x),
  hidden_dims = c(32, 16),
  output_dim = 16,
  dropout_rate = 0.5
)

image_mod <- image_module(
  in_channels = dim(x_image)[2],
  img_size = dim(x_image)[3],
  conv_channels = c(16, 32),
  kernel_size = 3,
  pool_kernel = 2,
  output_dim = 32
)

# Define the fusion network, MST-PMDN head, and training setup 
hidden_dim <- c(64, 32)
drop_hidden <- 0.5
n_mixtures <- 2
constraint <- "VVIFN"
fixed_nu <- c(50, NA)
constant_attr <- ""
wd_tabular <- 0
wd_image <- 0.2

# Combine the tabular module, image module, and fusion network
model <- define_mst_pmdn(
  input_dim = ncol(x),
  output_dim = ncol(y),
  hidden_dim = hidden_dim,
  n_mixtures = n_mixtures,
  image_module = image_mod,
  tabular_module = tabular_mod
)

# Model training
fit <- train_mst_pmdn(
  inputs = x,
  outputs = y,
  hidden_dim = hidden_dim,
  drop_hidden = drop_hidden,
  n_mixtures = n_mixtures,
  constraint = constraint,
  fixed_nu = fixed_nu,
  constant_attr = constant_attr,
  epochs = 10,
  lr = 1e-3,
  batch_size = 32,
  wd_tabular = wd_tabular,
  wd_image = wd_image,
  image_inputs = x_image,
  image_module = image_mod,
  tabular_module = tabular_mod,
  checkpoint_path = "wave_surge_checkpoint.pt",
  device = device
)

# Model inference
pred <- predict_mst_pmdn(
  fit$model,
  new_inputs = x,
  image_inputs = x_image,
  device = device
)
print(names(pred))
print(pred$pi[1:3, ])
print(pred$mu[1:3, , ])

# Draw samples 
df_samples <- sample_mst_pmdn_df(
  pred,
  num_samples = 1,
  device = device
)
print(head(df_samples))
```

## References 

Ambrogioni, L., Güçlü, U., van Gerven, M. A., & Maris, E. (2017). The kernel mixture network: A nonparametric method for conditional density estimation of continuous random variables. arXiv:1705.07111. 
 
Andrews, J. L., & McNicholas, P. D. (2012). Model-based clustering, classification, and discriminant analysis via mixtures of multivariate t-distributions: the t EIGEN family. Statistics and Computing, 22, 1021-1029. 
 
Azzalini, A., & Capitanio, A. (2003). Distributions generated by perturbation of symmetry with emphasis on a multivariate skew t-distribution. Journal of the Royal Statistical Society Series B: Statistical Methodology, 65(2), 367-389. 
 
Andrews, J. L., Wickins, J. R., Boers, N. M., & McNicholas, P. D. (2018). teigen: An R package for model-based clustering and classification via the multivariate t distribution. Journal of Statistical Software, 83, 1-32. 
 
Banfield, J. D., & Raftery, A. E. (1993). Model-based Gaussian and non-Gaussian clustering. Biometrics, 803-821. 
 
Celeux, G., & Govaert, G. (1995). Gaussian parsimonious clustering models. Pattern Recognition, 28(5), 781-793. 

Falbel D., & Luraschi, J. (2025). torch: Tensors and Neural Networks with 'GPU' Acceleration. R package version 0.14.2, https://github.com/mlverse/torch, https://torch.mlverse.org/docs.

Fraley, C., & Raftery, A. E. (2002). Model-based clustering, discriminant analysis, and density estimation. Journal of the American Statistical Association, 97(458), 611-631. 
 
Fraley, C., & Raftery, A. E. (1998). How many clusters? Which clustering method? Answers via model-based cluster analysis. The Computer Journal, 41(8), 578-588. 
 
Lee, S., & McLachlan, G. J. (2014). Finite mixtures of multivariate skew t-distributions: some recent and new results. Statistics and Computing, 24, 181-202. 

Kingma, D. P., & Ba, J. (2015). Adam: a method for stochastic optimization. Proceedings of the 3rd International Conference on Learning Representations, ICLR 2015, San Diego, CA, USA. arXiv:1412.6980 

Klein, N. (2024). Distributional regression for data analysis. Annual Review of Statistics and Its Application, 11:321-346.

Peel, D., & McLachlan, G.J. (2000). Robust mixture modelling using the t distribution. Statistics and Computing 10, 339–348. 

Srucca, L., Fop, M., Murphy, T. B., & Raftery, A. E. (2016). mclust 5: Clustering, classification and density estimation using Gaussian finite mixture models. The R Journal, 8(1), 289-317. 
 
Williams, P. M. (1996). Using neural networks to model conditional multivariate densities. Neural Computation, 8(4), 843-854.

---

# Deep MST-PMDN Architecture

<img src="deep-MST-PMDN.png" alt="Deep MST-PMDN" width="360"/>

## Function Summaries

The implementation consists of several key functions and modules:

### Function: `t_cdf(z, nu, nu_switch = 20)`

*   **Purpose:** Calculates a differentiable *approximation* of the univariate Student's t cumulative distribution function (CDF).
*   **Method:** For `nu >= nu_switch`, transforms the input quantile `z` using a scaling factor derived from the degrees of freedom `nu`, then computes the standard normal CDF of the result using the error function (`erf`). Otherwise, uses `pt` from R and manually inserts a gradient for `z` into the computation graph and uses a finite difference approximation for `nu`. Alternatively, can numerically integrate a torch-compatible probability density function `t_pdf_int`, which will be faster on GPUs.
*   **Context:** Switches between the fast approximation and the slow `pt` (or numerical integration) calculation. Essential for the loss function's skewness calculation.

### Function: `sample_gamma(shape, scale, device)`

*   **Purpose:** Generates random samples from a Gamma distribution using `torch`.
*   **Method:** Wraps R's `rgamma` function, vectorizes it using `mapply`, and converts the output to a `torch` tensor on the specified device.
*   **Context:** Used within the `sample_mst_pmdn` function to generate the scaling variable needed for sampling from the t-distribution component of the skew-t.

### Function: `build_orthogonal_matrix(params, dim)`

*   **Purpose:** Constructs a batch of orthogonal matrices (representing rotation/orientation `D`).
*   **Method:** Uses the matrix exponential of a skew-symmetric matrix, where the input `params` parameterize the upper triangle of the skew-symmetric matrix.
*   **Context:** Used in the main model (`define_mst_pmdn`) to generate the orientation component `D` of the LAD decomposition when orientation is not fixed to the identity matrix.

### Function: `init_mu_kmeans(model, outputs_train, ...)`

*   **Purpose:** Initializes the component mean parameters (`mu`) using k-means clustering.
*   **Method:** Applies k-means to the training output data to find initial centroids. These centroids initialize either the `model$mu` parameters (if constant) or the bias of the `model$fc_mu` layer (if network-dependent), setting initial weights to zero.
*   **Context:** A heuristic to provide a potentially better starting point for training compared to random initialization, aiming for faster convergence.

### Module: `weight_norm_linear` (nn_module)

*   **Purpose:** Implements a linear layer with weight normalization.
*   **Method:** Decomposes the weight matrix `W` into a direction `V` and a magnitude `g`, learning these instead of `W` directly.
*   **Context:** Used for most linear layers within the network architecture (hidden layers and parameter prediction heads) to potentially improve training stability and convergence speed.

### Function: `init_weight_norm(module)`

*   **Purpose:** Initializes the parameters (`V`, `g`) of a `weight_norm_linear` layer.
*   **Method:** Uses Kaiming (He) normal initialization for the direction `V` and sets the initial magnitude `g` accordingly.
*   **Context:** Applied recursively to the model to ensure proper initialization of all weight-normalized layers.

### Module: `define_mst_pmdn(...)` (nn_module)

*   **Purpose:** Defines the main MST-PMDN neural network architecture.
*   **Method:**
    *   Processes optional image and tabular inputs through dedicated modules or uses raw inputs.
    *   Concatenates features and passes them through a hidden MLP using `weight_norm_linear` layers.
    *   Predicts mixture parameters (`pi`, `mu`, `L`, `A`, `D`, `nu`, `alpha`) using separate output heads (mostly `weight_norm_linear` or `nn_parameter` if constant).
    *   Applies constraints (Variable, Equal, Identity, Normal approx., Fixed) to parameters based on configuration.
    *   Constructs the full scale matrix `Sigma = L * D * diag(A) * D^T` and computes its Cholesky decomposition (`scale_chol`) for each component.
*   **Output:** Returns a list containing all mixture parameters (`pi`, `mu`, `scale_chol`, `nu`, `alpha`) and LAD components (`L`, `A`, `D`), batched appropriately.

### Function: `loss_mst_pmdn(output, target)`

*   **Purpose:** Computes the negative log-likelihood (NLL) loss.
*   **Method:**
    *   For each data point and mixture component `k`:
        *   Calculates residuals: `diff = target - mu_k`.
        *   Standardizes residuals: `v = scale_chol_k^{-1} * diff`.
        *   Calculates squared Mahalanobis distance: `maha = ||v||^2`.
        *   Calculates the log-PDF of the symmetric multivariate t-distribution part using `maha`, `log_det(Sigma_k)`, and `nu_k`.
        *   Calculates the skewness adjustment term `log(2 * T_CDF(alpha_k^T w, df=nu_k+d))`, where `w` is proportional to `v`, using `t_cdf`.
    *   Combines component log-densities using mixture weights `pi` via `logsumexp`.
    *   Returns the mean NLL over the batch.

### Function: `sample_mst_pmdn(mdn_output, num_samples, ...)`

*   **Purpose:** On-device generation of random samples from the predicted mixture distribution.
*   **Method:**
    *   Samples component indices based on `pi`.
    *   Gathers parameters for the selected components.
    *   Generates t-distribution scaling factors `W` using `sample_gamma`.
    *   Generates a *standard* multivariate skew-normal sample `X` based on the component's `alpha` (via `delta`).
    *   Transforms the standard sample `X` to the output space: `Y = mu_s + W * (scale_chol_s @ X)`.
*   **Output:** Returns a **list** with  
    *  `samples` - a torch tensor of shape `[S, B, d]`, where `S` is `num_samples`, `B` is the batch size (rows of the predictor matrix), and `d` is the response dimension.  
    *  `components` - a torch tensor of shape `[S, B]` giving the **1-based** component label (`1..G`) used for each draw.

### Function: `sample_mst_pmdn_df(mdn_output, num_samples, ...)`

*   **Purpose:** Generates random samples from the predicted mixture distribution and returns a formatted R data frame.
*   **Method:**
    *   Samples component indices based on `pi`.
    *   Gathers parameters for the selected components.
    *   Generates t-distribution scaling factors `W` using `sample_gamma`.
    *   Generates a *standard* multivariate skew-normal sample `X` based on the component's `alpha` (via `delta`).
    *   Transforms the standard sample `X` to the output space: `Y = mu_s + W * (scale_chol_s @ X)`.
*   **Output:** A data frame with `num_samples * batch_size` rows containing  
    *  simulated response variables in columns `V1 ... Vd`;  
    *  `row` - the index (`1..B`) of the predictor row that generated the draw;  
    *  `draw` - the draw number (`1..num_samples`) for that predictor row;  
    *  `comp` - a factor giving the 1-based component label (`1..G`).  

### Function: `train_mst_pmdn(...)`

*   **Purpose:** Manages the model training process.
*   **Method:** Includes data loading, model/optimizer setup (with k-means init), training loop (loss calculation, backpropagation, optimization), validation, learning rate scheduling, checkpointing, and early stopping. Handles optional image inputs correctly.
*   **Output:** Trained model, loss history, and training/validation indices.

### Function: `predict_mst_pmdn(model, new_inputs, ...)`

*   **Purpose:** Performs inference using the trained model.
*   **Method:** Runs a forward pass on new inputs in evaluation mode (`torch_no_grad()`).
*   **Output:** Raw model output list containing mixture parameters for the new inputs.
