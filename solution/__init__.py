import joblib
import os
import pandas as pd
import sklearn.metrics
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
import sys
import numpy as np
import tensorflow as tf
from sklearn import preprocessing
from sklearn.utils.multiclass import unique_labels
from sklearn.feature_extraction import DictVectorizer
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction import FeatureHasher
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import FunctionTransformer



# Feature columns describe how to use the input.




def ds_transform(ds,y=None):
    del ds['native-country']
    ds = ds.replace({' ?': np.nan})
    ds = pd.get_dummies(ds)
    return ds

def ds_scaler(ds,y=None):
    scaler = MinMaxScaler()
    ds[ds.columns] = scaler.fit_transform(ds[ds.columns])
    return ds




#
#This function should build an sklearn.pipeline.Pipeline object to train
#and evaluate a model on a pandas DataFrame. The pipeline should end with a
#custom Estimator that wraps a TensorFlow model. See the README for details.
#
def get_pipeline():
    #First transformers, one hot encodes the non continuous features
    transformer_1 = FunctionTransformer(ds_transform,validate=False)

    #Second transformer, scales the panda dataset using an sklearn minmaxscaler
    transformer_2 = FunctionTransformer(ds_scaler,validate=False)

    #make pipeline
    pipe = make_pipeline(transformer_1,transformer_2,DNNEstimator())
    return pipe


class DNNEstimator(BaseEstimator):
    """ A template estimator to be used as a reference implementation .
    Parameters
    ----------
    demo_param : str, optional
        A parameter used for demonstation of how to pass and store paramters.
    """
        


    def fit(self, X, y):
        """A reference implementation of a fitting function
        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The training input samples.
        y : array-like, shape = [n_samples] or [n_samples, n_outputs]
            The target values (class labels in classification, real numbers in
            regression).
        Returns
        -------
        self : object
            Returns self.
        """
        dict_f = dict(X)
        for key in dict(X).keys():
            dict_f[key.replace(" ", "").replace("?", "").replace("(", "").replace(")", "").replace("&","")] = dict_f.pop(key)
        my_feature_columns = []
        for key in dict_f.keys():
            my_feature_columns.append(tf.feature_column.numeric_column(key=key))
        self.classifier = tf.estimator.DNNClassifier(feature_columns=my_feature_columns,hidden_units=[20, 20,20],n_classes=2)



        def train_input_fn(features, labels, batch_size):
            dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
            dataset = dataset.shuffle(buffer_size=1000).repeat(count=None).batch(batch_size)
            return dataset.make_one_shot_iterator().get_next()




        self.classifier.train(
            input_fn=lambda:train_input_fn(dict_f, y.values, 100),
            steps=1000)


        # Return the estimator
        return self

    def predict(self, X):
        """ A reference implementation of a predicting function.
        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.
        Returns
        -------
        y : array of shape = [n_samples]
            Returns :math:`x^2` where :math:`x` is the first column of `X`.
        """
        def eval_input_fn(features, labels=None, batch_size=None):
            """An input function for evaluation or prediction"""
            if labels is None:
                # No labels, use only features.
                inputs = features
            else:
                inputs = (features, labels)

            # Convert inputs to a tf.dataset object.
            dataset = tf.data.Dataset.from_tensor_slices(inputs)

            # Batch the examples
            assert batch_size is not None, "batch_size must not be None"
            dataset = dataset.batch(batch_size)

            # Return the read end of the pipeline.
            return dataset.make_one_shot_iterator().get_next()

        dict_f = dict(X)
        for key in dict(X).keys():
            dict_f[key.replace(" ", "").replace("?", "").replace("(", "").replace(")", "").replace("&","")] = dict_f.pop(key)
        predictions = self.classifier.predict(input_fn=lambda:eval_input_fn(dict_f,labels=None,batch_size=10))
        preds = np.zeros((X.shape[0], 2))
        i=0
        for pred_dict in predictions:
            preds[i,0:2] = pred_dict['probabilities']

            i += 1
        return preds


#Function to transform Dataset into numpy array, using sklearn transform
