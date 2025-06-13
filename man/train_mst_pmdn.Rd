\name{train_mst_pmdn}
\alias{train_mst_pmdn}
\title{Train a Deep Multivariate Skew t Parsimonious Mixture Density Network}
\description{
Fits an MST-PMDN model by minimizing the negative log-likelihood on the supplied data, with support for optional image inputs, checkpointing, early stopping, gradient clipping, weight decay, dropout, and learning-rate scheduling.  
}
\usage{
train_mst_pmdn(
  inputs,
  outputs,
  hidden_dim,
  n_mixtures,
  constraint             = "VVVNN",
  constant_attr          = "",
  fixed_nu               = NULL,
  range_nu               = c(3, 50),
  nu_switch              = 20,
  max_alpha              = 5,
  min_vol_shape          = 1e-2,
  min_mix_weight         = 1e-4,
  jitter                 = 1e-6,
  activation             = nn_tanh,
  epochs                 = 500,
  lr                     = 0.001,
  batch_size             = 16,
  max_norm               = 1,
  drop_hidden            = 0,
  wd_image               = 0,
  wd_tabular             = 0,
  checkpoint_interval    = 10,
  checkpoint_path        = "checkpoint.pt",
  resume_from_checkpoint = FALSE,
  model                  = NULL,
  early_stopping_patience= 50,
  validation_split       = 0.2,
  custom_split           = NULL,
  scheduler_step         = 50,
  scheduler_gamma        = 0.5,
  image_inputs           = NULL,
  image_module           = NULL,
  tabular_module         = NULL,
  device                 = "cpu"
)
}
\arguments{
  \item{inputs}{Numeric matrix or \code{\link[torch]{torch_tensor}} of predictors (rows = cases).}
  \item{outputs}{Numeric matrix or \code{\link[torch]{torch_tensor}} of responses (rows = cases).}
  \item{hidden_dim}{Integer vector giving the sizes of the hidden layers in the fusion network.}
  \item{n_mixtures}{Integer; number of mixture components \code{M}.}
  \item{constraint}{Character; MST-PMDN constraint code (volume–shape–orientation, degrees of freedon \code{nu}, and skew), e.g. \code{"VVVFN"}.}
  \item{constant_attr}{Character; flags for parameters held constant ("m" for means, "x" for mixing coefficients, "L" for volume, "A" for shape, "D" for orientation, "n" for degrees of freedom \code{nu}, and "s" for skew).}
  \item{fixed_nu}{Numeric vector of length \code{M}, or \code{NULL}; if non-\code{NULL}, fixes degrees of freedom \code{nu} per component.}
  \item{range_nu}{Numeric vector of length 2: clamp range for \code{nu}.}
  \item{nu_switch}{Numeric; \code{nu} threshold for switching between fast vs slow CDF routines.}
  \item{max_alpha}{Numeric; maximum absolute skewness parameter.}
  \item{min_vol_shape}{Numeric; minimum allowed volume (L) and shape (A) diagonal values.}
  \item{min_mix_weight}{Numeric; minimum mixture weight per component.}
  \item{jitter}{Numeric; small ridge added to covariance diagonals for stability.}
  \item{activation}{Activation function for hidden layers (e.g., \code{nn_tanh}, \code{nn_relu}).}
  \item{epochs}{Integer; maximum number of training epochs.}
  \item{lr}{Numeric; learning rate for the optimizer.}
  \item{batch_size}{Integer; mini-batch size.}
  \item{max_norm}{Numeric; maximum gradient norm for clipping.}
  \item{drop_hidden}{Numeric in [0,1]; dropout probability after each hidden layer.}
  \item{wd_image}{Numeric; weight decay (L2 penalty) for \code{image_module} parameters.}
  \item{wd_tabular}{Numeric; weight decay for \code{tabular_module} parameters.}
  \item{checkpoint_interval}{Integer; number of epochs between saved checkpoints.}
  \item{checkpoint_path}{Character; file path for saving model checkpoints.}
  \item{resume_from_checkpoint}{Logical; if \code{TRUE}, loads and resumes from an existing checkpoint.}
  \item{model}{An existing \code{mst_pmdn_model} to continue training, or \code{NULL} to initialize a new one.}
  \item{early_stopping_patience}{Integer; epochs with no validation improvement before stopping.}
  \item{validation_split}{Numeric in [0,1]; fraction of cases held out for validation if \code{custom_split} is \code{NULL}.}
  \item{custom_split}{Index or logical vector, or list with \code{train} and \code{validation} elements; overrides \code{validation_split}.}
  \item{scheduler_step}{Integer; epochs between learning-rate decay steps.}
  \item{scheduler_gamma}{Numeric; multiplicative factor for learning-rate decay.}
  \item{image_inputs}{Optional array or \code{torch_tensor} of image data (4D) for multimodal models.}
  \item{image_module}{Optional \code{nn_module} for processing \code{image_inputs}.}
  \item{tabular_module}{Optional \code{nn_module} for processing \code{inputs}.}
  \item{device}{Character; torch device to use (e.g., \code{"cpu"} or \code{"cuda"}).}
}
\details{
Splits the data into training/validation sets (unless \code{custom_split} is provided), initializes or loads the model,
and then runs an Adam optimization loop with optional gradient clipping, weight decay, dropout, early stopping, checkpointing,
and learning-rate scheduling.  After training, the best‐performing model (by validation or training loss) is reloaded before returning. 
}
\value{
A \code{list} with components:
\describe{
  \item{model}{The trained \code{mst_pmdn_model} (torch module).}
  \item{train_loss_history}{Numeric vector of training losses per epoch.}
  \item{val_loss_history}{Numeric vector of validation losses per epoch, or \code{NULL} if none.}
  \item{best_train_epoch}{Epoch number with lowest training loss (if no validation split).}
  \item{best_train_loss}{Lowest training loss achieved (if no validation split).}
  \item{best_val_epoch}{Epoch number with lowest validation loss (if used).}
  \item{best_val_loss}{Lowest validation loss achieved (if used).}
  \item{final_epoch}{Last completed epoch.}
  \item{train_indices}{Indices of cases used for training.}
  \item{val_indices}{Indices of cases used for validation.}
}
}
