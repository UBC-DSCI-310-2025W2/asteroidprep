# asteroidprep

`asteroidprep` is a Python package for our asteroid analysis project.

## What it does

This package contains functions for:

- cleaning asteroid data,
- dropping unnecessary columns,
- generating a KNN parameter grid,
- preparing evaluation data, and
- splitting predictors and target variables.

## Installation

Install the package from the `v0.1.2` release tag with:

```bash
pip install "git+https://github.com/UBC-DSCI-310-2025W2/asteroidprep.git@v0.1.2"
```

## Usage

Import the package functions with:

```
from asteroidprep import (
    clean_pha,
    clean_full_name,
    drop_columns,
    get_knn_param_grid,
    prepare_eval_data,
    split_features_target,
)
```

## Main project

This package supports the analysis repository for the asteroid project.
