################################################################################

rm(list=ls())
source("../MST-PMDN.R")
torch_set_num_threads(1)

sample_skew_mixture <- function(n = 1000, seed = NULL) {
  # Set seed for reproducibility if provided
  if (!is.null(seed)) {
    set.seed(seed)
  }
  
  # Define mixture weights
  weights <- c(0.7, 0.3)  # 70% main component, 30% heavy-tailed component
  
  # Number of samples from each component
  n1 <- round(n * weights[1])
  n2 <- n - n1
  
  # Component 1: Main concentration (using skew-normal)
  xi1 <- c(-0.5, 0.5)       # Location parameters
  Omega1 <- matrix(c(0.1, 0, 
                     0, 1), 2, 2)  # Scale matrix
  alpha1 <- c(-1, 2)     # Shape parameters controlling skewness
  
  # Component 2: Heavy-tailed component extending to the right (using skew-t)
  xi2 <- c(2, 0)         # Location shifted to the right
  Omega2 <- matrix(c(1.2, 0.3, 
                     0.3, 0.4), 2, 2)  # Different scale
  alpha2 <- c(3, -0.5)   # Shape parameters for rightward skew
  nu1 <- 30              # Degrees of freedom for tail1
  nu2 <- 7               # Degrees of freedom for tail2
  
  # Generate samples from each component
  samples1 <- sn::rmst(n1, xi = xi1, Omega = Omega1, alpha = alpha1, nu = nu1)
  samples2 <- sn::rmst(n2, xi = xi2, Omega = Omega2, alpha = alpha2, nu = nu2)
  
  # Combine samples
  result <- rbind(samples1, samples2)
  
  # Column names for clarity
  colnames(result) <- c("x", "y")
  
  return(result)
}

# ------------------------------
# Create Synthetic Data with Mixture of 2 Components
# ------------------------------

# Number of samples
n_samples <- 3000

# Tabular input: e.g., 10 features per sample
tabular_input_dim <- 10
inputs <- torch_randn(n_samples, tabular_input_dim)

# Image inputs: Grayscale images, 28x28 (1 channel).
image_inputs <- torch_randn(n_samples, 1, 28, 28)

# Outputs
output_matrix <- scale(sample_skew_mixture(n_samples))
outputs <- torch_tensor(output_matrix, dtype=torch_float())

# Create a weak relationship with inputs to maintain some predictability
tabular_effect <- 0.2 * inputs$sum(dim = 2, keepdim = TRUE)
image_feature <- 0.2 * image_inputs$view(c(n_samples, -1))$mean(dim = 2, keepdim = TRUE)

# Add a small input-dependent effect to the outputs
outputs[, 1] <- outputs[, 1] + tabular_effect$squeeze() + image_feature$squeeze()
outputs[, 2] <- outputs[, 2] + 0.5 * tabular_effect$squeeze() + 0.8 * image_feature$squeeze()

# ------------------------------
# Define a Simple Tabular Module
# ------------------------------
# This module processes the tabular inputs into a feature representation.
tabular_module <- nn_module(
  "TabularModule",
  initialize = function(input_dim, hidden_dim, output_dim) {
    self$fc1 <- nn_linear(input_dim, hidden_dim)
    self$fc2 <- nn_linear(hidden_dim, output_dim)
    self$output_dim <- output_dim
  },
  forward = function(x) {
    x <- nnf_relu(self$fc1(x))
    x <- self$fc2(x)
    x
  }
)

# Instantiate the tabular module.
# For example: input dimension 10, hidden layer of size 8, output feature dimension 8.
tabular_mod <- tabular_module(input_dim = tabular_input_dim, hidden_dim = 8, output_dim = 8)

# ------------------------------
# Define a Simple Image Module
# ------------------------------
# This module processes 28x28 grayscale images.
image_module <- nn_module(
  "ImageModule",
  initialize = function() {
    # One convolution: input channels=1, output channels=4, kernel size=3
    self$conv <- nn_conv2d(in_channels = 1, out_channels = 4, kernel_size = 3)
    # Max pooling with kernel size 2
    self$pool <- nn_max_pool2d(kernel_size = 2)
    
    # After conv: (28 - 3 + 1 = 26), then pooling: floor(26/2)=13.
    # Final flattened size: 4 (channels) * 13 * 13.
    self$flatten_dim <- 4 * 13 * 13
    # Linear layer to produce image feature vector of size 8.
    self$fc <- nn_linear(self$flatten_dim, 8)
    
    self$output_dim <- 8
  },
  forward = function(x) {
    x <- self$conv(x)
    x <- nnf_relu(x)
    x <- self$pool(x)
    x <- torch_flatten(x, start_dim = 2)
    x <- self$fc(x)
    nnf_relu(x)
  }
)

# Instantiate the image module.
image_mod <- image_module()

# ------------------------------
# Set MST-PMDN Hyperparameters and Train the Model
# ------------------------------

# Hidden layers configuration for the main MLP (after concatenating features).
hidden_dim <- 7

# Number of mixture components, LADns constraint, and stationary parameters
n_mixtures <- 2
constraint <- "VVVVV"
constant_attr <- "LADmxns"
fixed_nu <- c(100, NA)

# Training hyperparameters
epochs <- 100           # For demonstration, use a small number of epochs.
lr <- 0.001             # Learning rate
batch_size <- 16        # Batch size
wd_image <- 1e-6        # Image encoder weight decay
wd_tabular <- 1e-6      # Tabular encoder weight decay
wd_hidden <- 1e-5       # PMDN hidden layer weight decay

# Train the model.
# The train_mst_pmdn function handles data conversion, splitting, and checkpointing.
model_fit <- train_mst_pmdn(
  inputs = inputs,
  outputs = outputs,
  hidden_dim = hidden_dim,
  activation = nn_relu,
  range_nu = c(3., 50.),
  max_alpha = 5.,
  min_vol_shape = 1e-3,
  jitter = 1e-2,
  n_mixtures = n_mixtures,
  epochs = epochs,
  lr = lr,
  batch_size = batch_size,
  wd_hidden = wd_hidden,
  wd_image = wd_image,
  wd_tabular = wd_tabular,
  early_stopping_patience = 50,
  validation_split = 0.2,
  image_inputs = image_inputs,
  image_module = image_mod,
  tabular_module = tabular_mod,
  constraint = constraint,
  constant_attr = constant_attr,
  fixed_nu = fixed_nu,
  resume_from_checkpoint = FALSE,
  device = "cpu"
)

cat("Training completed at epoch:", model_fit$final_epoch, "\n")
if (!is.null(model_fit$best_val_loss)) {
  cat("Best validation loss:", model_fit$best_val_loss, "\n")
} else {
  cat("Best training loss:", model_fit$best_train_loss, "\n")
}

# ------------------------------
# Predict Using the Trained Model
# ------------------------------
# Here we reuse the same synthetic data for prediction.
predictions <- predict_mst_pmdn(model_fit$model, new_inputs = inputs,
                                image_inputs = image_inputs)

# The predictions are a list of mixture parameters (pi, mu, scale_chol, nu, alpha, and LAD components).
cat("Prediction structure:\n")
str(predictions)

# ------------------------------
# Sample from the Fitted Model
# ------------------------------
# Generate 5 new samples from the model’s predicted distribution.
samples <- sample_mst_pmdn(predictions, num_samples = 5, device = "cpu")

# samples has shape: [num_samples, batch_size, output_dim]
samples_shape <- samples$size()
cat("Shape of generated samples (num_samples, batch_size, output_dim):\n")
print(samples_shape)

# Show the generated samples from the first sample across all batches.
cat("Generated samples from the first sample (across batches):\n")
print(as_array(samples[1, , ]))

# -----------------------------------------------------------
# Print predictions/cor(ensemble mean of samples, outputs)
# -----------------------------------------------------------

print(predictions)
ens_mean <- apply(as.array(samples), c(2, 3), mean) # [n, d]
print(cor(ens_mean, as.matrix(outputs)))

# -------------------
# Diagnostic plots
# -------------------

pairs(as.matrix(outputs), pch=19, col=rgb(0,0,1,0.2))
pairs(as.array(samples)[1,,], pch=19, col=rgb(0,0,1,0.2))

plot(model_fit$train_loss_history, type='l', col=2, lwd=2,
     xlab = 'Epoch', ylab = 'Neg. log likelihood',
     ylim=range(pretty(c(model_fit$train_loss_history,
                         model_fit$val_loss_history))))
if(!is.null(model_fit$val_loss_history)){
  lines(model_fit$val_loss_history, col=4, lwd=2)
  abline(v=model_fit$best_val_epoch, lty=2, col=4, lwd=2)
}
legend("topright", legend=c("Train", "Validation"), col=c(2,4), lwd=2)

ens_mean <- apply(as.array(samples), c(2, 3), mean) # [n, d]
par(mfrow = c(1, 2))
plot(ens_mean[,1], as.matrix(outputs)[,1],
     xlab = "Predicted mean 1", ylab = "True Output 1", pch=19,
     col=rgb(0,0,1,0.4))
abline(0,1,lty=2)
plot(ens_mean[,2], as.matrix(outputs)[,2],
     xlab = "Predicted mean 2", ylab = "True Output 2", pch=19,
     col=rgb(0,0.7,0,0.4))
abline(0,1,lty=2)

################################################################################
