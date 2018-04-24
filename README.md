# RADIX.AI Machine Learning challenge

## Introduction

The goal of this challenge is to build an sklearn.pipeline.Pipeline that chains the necessary sklearn Estimators to train and evaluate a model on a pandas DataFrame of the [Adult Data Set](http://mlr.cs.umass.edu/ml/datasets/Adult). The resulting model will predict if an adult's yearly income is above 50K USD or not.

## Getting started

1. Clone this repository (do not fork it!) and upload it to a fresh repository that you create.
2. Install [Miniconda](https://conda.io/miniconda.html) if you don't have it already.
3. Run `conda env create` from the repo's base directory to create the repo's conda environment from `environment.yml`. You may add packages listed on [anaconda.org](https://anaconda.org/) to `environment.yml` as desired.
4. Run `activate machine-learning-challenge-env` to activate the conda environment.
5. Start implementing the `def pipeline():` function in the `solution` directory!

## Evaluating your solution

To check your solution, run `python challenge.py` from the base of this repository. This will trigger the following steps:

1. Call `fitted_pipeline = solution.pipeline().fit(X_train, y_train)` where `X_train` is a pandas DataFrame and `y_train` is a pandas Series of labels.
2. Call `y_pred = fitted_pipeline.predict_proba(X_test)` where `X_test` is a pandas DataFrame of the same format as `X_train`.
3. Compute the ROC AUC between `y_pred` and `y_test` and print your score!
4. When you're ready, send us the URL to your repo!

Good luck!

-- radix.ai
