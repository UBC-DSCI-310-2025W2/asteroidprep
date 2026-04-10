# asteroidprep

`asteroidprep` is a Python package for our asteroid analysis project.

## What it does

This package contains functions for:

- cleaning asteroid data,
- dropping unnecessary columns,
- generating a KNN parameter grid,
- preparing evaluation data, and
- splitting predictors and target variables.

Additional functions are available for data validation, exploratory data analysis, and selecting classification thresholds based on predicted probabilities.

## Installation

Install the package from the `v0.1.2` release tag with:

```bash
pip install "git+https://github.com/UBC-DSCI-310-2025W2/asteroidprep.git@v0.1.2"
```

## Usage

Import the main package functions with:

```python
from asteroidprep import (
    clean_pha,
    clean_full_name,
    drop_columns,
    get_knn_param_grid,
    prepare_eval_data,
    split_features_target,
)
```

Import automatic plot scale inference with:

```python
from asteroidprep.infer_plot_scales import infer_plot_scales
```

Import the data validation functions with:

```python
from asteroidprep.validation import run_validation
```

Import threshold utilities with:

```python
from asteroidprep.threshold_utils import select_threshold
```

## Main project

This package supports the analysis repository for the asteroid project.
