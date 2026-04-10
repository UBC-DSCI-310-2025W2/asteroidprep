# asteroidprep

Authors: Jerry Jin, Malcolm Maxwell, Sadie Lee
[![CI](https://github.com/UBC-DSCI-310-2025W2/asteroidprep/actions/workflows/ci-cd.yml/badge.svg?branch=main)](https://github.com/UBC-DSCI-310-2025W2/asteroidprep/actions/workflows/ci-cd.yml)

## About

This Python package is directly intended for our asteroid project that aims to predict potentially hazardous near-earth asteroids, containing a few helper functions that aid in ensuring a reproducible workflow. In our main project we build a k-nearest neighbors algorithm using features from NASA JPL's Small-Body DataBase (SBDB). Specifically, this package separates our preprocessing functions from this main analysis repository so that the analysis can import these functions as a package rather than keeping them directly inside the report repository. Documentation of this database can be found at https://ssd-api.jpl.nasa.gov/doc/sbdb_query.html.

## Installation

Install the package from the `v0.1.0` release tag with:

```bash
pip install "git+https://github.com/UBC-DSCI-310-2025W2/asteroidprep.git@v0.1.0"
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

## What this package contains

This package includes the following `py` files, containing respective functions:

- `clean.py` aims to clean our original data,
- `drop_columns.py` aids in dropping unnecessary columns,
- `get_knn_param_grid.py` generates a KNN parameter grid,
- `prepare_eval_data.py` prepares our data for evaluation, and
- `split_features_target.py` splits our predictors and target variables.

Together these functions support the preprocessing steps used in our workflow.

## Where this package sits in the Python ecosystem

This small package, `asteroidprep`, is intended to support our asteroid analysis project's workflow, and is meant to be utilized along with larger packages like `pandas` and `scikit-learn`. In particular, its functionality makes our data preparation easier. In other words, it is NOT meant to replace the typical larger packages that Python offers.

## Our main repository

The asteroid analysis project that this package is based on can be found at:
https://github.com/UBC-DSCI-310-2025W2/dsci-310-group-18

## License

This project is offered under the Attribution 4.0 International (CC BY 4.0) License. The software in this project is offered under the MIT open source license. See the license file for more information.
