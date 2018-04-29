import pandas as pd
from sklearn.pipeline import make_pipeline
import numpy as np
import tensorflow as tf
from sklearn.externals import joblib
from sklearn.utils.multiclass import unique_labels
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import FunctionTransformer






def ds_transform(ds, y=None):
    """Transforms the panda dataframe into one with only continuous features"""
    #changing native-country so it is able to use that data, but only as
    #wether the person is from the US or not, too little data otherwise
    ds = ds.rename(columns={'native-country': 'nativecountry'})
    ds.nativecountry[ds.nativecountry.str.contains('United-States')] = 0
    ds.nativecountry[ds.nativecountry != 0] = 1
    #replacing ? with NaN this prevents them from becoming a seperate categorical
    #feature
    ds = ds.replace({' ?': np.nan})
    #Creating categorical/binary features from the strings
    ds = pd.get_dummies(ds)
    return ds

def ds_scaler(ds, y=None):
    """Scales the panda DataFrame"""
    #scaling age,income etc to a value between [0,1], so they do not
    #get extra weight compared to the binary features
    scaler = MinMaxScaler()
    ds[ds.columns] = scaler.fit_transform(ds[ds.columns])
    return ds





#This function should build an sklearn.pipeline.Pipeline object to train
#and evaluate a model on a pandas DataFrame. The pipeline should end with a
#custom Estimator that wraps a TensorFlow model. See the README for details.
def get_pipeline():
    """Builds a pipeline of 2 transformers and a tensoflow Estimator

    Returns
    -------
    Pipeline object
    """

    #First transformer, one hot encodes the non continuous features
    transformer_1 = FunctionTransformer(ds_transform, validate=False)

    #Second transformer, scales the panda dataset using an sklearn minmaxscaler
    transformer_2 = FunctionTransformer(ds_scaler, validate=False)

    #make pipeline
    pipe = make_pipeline(transformer_1, transformer_2, DNNEstimator())

    #Saves pipeline: Does not save the custom fuctions used in the custom
    #estimator
    joblib.dump(pipe, 'pipeline.pkl')
    return pipe


class DNNEstimator(BaseEstimator):
    """ A template estimator to be used as a reference implementation .
    Parameters
    ----------
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
        #tensorflow does not accept some characters as feature names
        for key in dict(X).keys():
            dict_f[key.replace(" ", "").replace("?", "").replace(
                "(", "").replace(")", "").replace("&", "")] = dict_f.pop(key)
        my_feature_columns = []
        for key in dict_f.keys():
            my_feature_columns.append(tf.feature_column.numeric_column(key=key))
        #Creating a DNN clasifier
        self.classifier = tf.estimator.DNNClassifier(
            feature_columns=my_feature_columns,
            hidden_units=[25, 25], n_classes=2)

        #input_fn as defined in the estimator guidefor tensorflow
        def train_input_fn(features, labels, batch_size):
            """Input function for training"""
            dataset = tf.data.Dataset.from_tensor_slices((features, labels))
            dataset = dataset.shuffle(buffer_size=1000).repeat(count=None).batch(batch_size)
            return dataset.make_one_shot_iterator().get_next()

        #Create classifier with a batch size of 100
        self.classifier.train(
            input_fn=lambda: train_input_fn(dict_f, y.values, 50),
            max_steps=2000)


        # Return the estimator
        return self

    def predict_proba(self, X):
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

        #input_fn as defined in the estimator guide of tensorflow
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
        #Tensorflow does not accept some characters as feature names
        for key in dict(X).keys():
            dict_f[key.replace(" ", "").replace("?", "").replace(
                "(", "").replace(")", "").replace("&", "")] = dict_f.pop(key)
        #predict the given inputs: returns a generator object of dicts
        predictions = self.classifier.predict(input_fn=lambda: eval_input_fn(
            dict_f, labels=None, batch_size=10))
        preds = np.zeros((X.shape[0], 2))
        i = 0
        #put predictions in the right shape as requested by the challenge.py
        for pred_dict in predictions:
            preds[i, 0:2] = pred_dict['probabilities']
            i += 1
        return preds
