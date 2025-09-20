# PyTorch MST-PMDN Companion

This directory provides a self-contained PyTorch implementation of the
Multivariate Skew-t Parsimonious Mixture Density Network (MST-PMDN).  The code
follows the architecture and parameterisation of the original R package while
replacing the Student-*t* helpers with differentiable `torch.distributions`
primitives.

## Layout

```
python/
├── README.md                 # This document
├── pyproject.toml            # Packaging metadata for editable installs
├── src/mst_pmdn/             # Library sources
│   ├── __init__.py
│   ├── distributions.py      # Student-t CDF helpers
│   ├── losses.py             # Negative log-likelihood
│   ├── model.py              # MST-PMDN head
│   ├── modules.py            # Weight-normalised linear layer
│   ├── sampling.py           # Posterior sampling utilities
│   ├── train.py              # Minimal training loop helper
│   └── utils.py              # Miscellaneous helpers (gamma sampling, k-means, ...)
└── tests/                    # PyTest-based unit tests
    ├── __init__.py
    ├── test_distributions.py
    ├── test_model.py
    ├── test_sampling.py
    ├── test_train.py
    └── test_utils.py
```

## Running the tests

Create a virtual environment, install the package in editable mode and run the
`pytest` suite:

```bash
cd python
python -m venv .venv
source .venv/bin/activate
pip install -e .
pytest
```

The test-suite exercises the utility helpers, verifies that the Student-*t* CDF
uses PyTorch without falling back to approximations, checks output shapes and
constraints, confirms sampling works on-device, and runs a miniature training
loop to validate gradient flow.
