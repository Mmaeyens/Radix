# Machine Learning challenge

## Introduction

The goal of this challenge is to build a Machine Learning model to predict if a given adult's yearly income is above or below $50k.

To succeed, you must develop a `solution` Python package that implements a `get_pipeline` function that returns:

- [x] an [sklearn.pipeline.Pipeline](http://scikit-learn.org/stable/modules/pipeline.html) that
- [x] chains a series of [sklearn Estimators](http://scikit-learn.org/stable/data_transforms.html) of your choice, and must end with
- [x] a [TensorFlow model](https://www.tensorflow.org/get_started/custom_estimators) wrapped as a [custom sklearn Estimator](http://scikit-learn.org/stable/developers/contributing.html#rolling-your-own-estimator), and will be fed
- [x] a pandas DataFrame of the [Adult Data Set](http://mlr.cs.umass.edu/ml/datasets/Adult) to train and evaluate the pipeline.

## Getting started

1. Clone this repository (do not fork it!) and upload it to a fresh repository that you create.
2. Install [Miniconda](https://conda.io/miniconda.html) if you don't have it already.
3. Run `conda env create` from the repo's base directory to create the repo's conda environment from `environment.yml`. You may add packages listed on [anaconda.org](https://anaconda.org/) to `environment.yml` as desired.
4. Run `activate machine-learning-challenge-env` to activate the conda environment.
5. Start implementing the `def get_pipeline():` function in the `solution` directory!

## Evaluating your solution

To check your solution, run `python challenge.py` from the base of this repository. This will trigger the following steps:

1. Call `fitted_pipeline = solution.get_pipeline().fit(X_train, y_train)` where `X_train` is a pandas DataFrame and `y_train` is a pandas Series of labels.
2. Call `y_pred = fitted_pipeline.predict_proba(X_test)` where `X_test` is a pandas DataFrame of the same format as `X_train`.
3. Compute the ROC AUC between `y_pred` and `y_test` and print your score!
4. When you're ready, send us the URL to your repo!

Good luck!

-- radix.ai
