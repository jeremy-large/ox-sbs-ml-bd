import logging
import numpy as np
import pylab as plt

import itertools

from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn import metrics

from uci_retail_data import stock_codes
from uci_retail_data import uci_files


def get_standard_data():
    df = uci_files.standard_uci_data_access()
    invalids = stock_codes.invalid_series(df)
    return df, invalids, stock_codes.invoice_df(df, invalid_series=invalids)


def __select_metric(y):
    if y.dtype == float:
        return metrics.mean_squared_error
    return metrics.confusion_matrix


def train_n_test(X, y, n_folds, update_frequency=None, model=None, metric=None, train_on_minority=False, concise=True):
    """
    @param X: a pandas DataFrame of features data of shape (A, B)
    @param y: a pandas Series of target data, of length A
    @param n_folds: the number of splits, or folds, of the data that we would like performed
    @param update_frequency: after implementing this many folds, provide an update
    @param model: by default LinearRegression(). Can also be set to another model with methods .fit() and .predict()
    @param metric: by default metrics.mean_squared_error. Can also be set to another metric
    @param train_on_minority: if set to True, then reverse the roles of test and train
    @return : a list of floats, each is the test MSE from a fold of the data
    """
    y = y.values.ravel()            # strip out from y just its float values as an array
    Xm = X.reset_index(drop=True)   # apply a new index to X, if necessary, which just counts up from 0 to (len(X)-1)

    update_frequency = update_frequency or len(y) - 1
    model = LinearRegression() if model is None else model
    metric = __select_metric(y) if metric is None else metric
    if type(n_folds) == int:
        kfold = model_selection.KFold(n_splits=n_folds, shuffle=True)

    scores = []
    for (train, test) in kfold.split(Xm, y):

        if train_on_minority:
            train, test = (test, train)    # swap em!

        if len(scores) % update_frequency == 0:
            if not concise:
                logging.info(
                    f"In study {len(scores) + 1}/{n_folds}, train on {len(train)} randomly selected points; "
                    f"then test on the other {len(test)}: first few test points = {test[:5]} ")
            else:
                logging.info(f"Study {len(scores) + 1}/{n_folds}: {len(train)} train rows;  {len(test)} test rows")

        model.fit(Xm.loc[train], y[train])

        score = metric(                              # r2_score(), and similar metrics, takes two arguments ...
                       y[test],                 # the actual targets, and ...
                       model.predict(Xm.loc[test])  # the fitted targets.
                      )                              # We get a number (maybe between 0 and 1) back

        scores.append(score)

    return scores


def plot_kfold_scores(ss, scatter=False, metric_name=None, n_bins=30):
    """
    @param ss: a list or array of quality data from a series of statistical fits
    @param scatter: Boolean. If True, plot a scatter of all the data, otherwise plot a histogram
    @param metric_name: an optional string which is the name of the metric, a list of which is in x
    @param n_bins: the number of bins to apply if a histogram is plotted
    @return : nothing
    """

    metric_name = metric_name or 'MSE'

    if scatter:
        plt.scatter(range(len(ss)), ss, marker='.')
        plt.axhline(np.mean(ss), color='r')
    else:
        plt.hist(ss, bins=n_bins)
        plt.axvline(np.mean(ss), color='r')

    plt.xlabel(f'Average value of the data, {np.round(np.mean(ss), 3)}, is shown in red')
    plt.title(f"{metric_name}, as calculated *only* on the testing datapoints from {len(ss)} different k-fold splits")

    plt.grid(); plt.axhline(0, color='k'); plt.show()


def build_polynomial_dataframe(data, der):
    """
    @param data: a DataFrame() of features (or regressors)
    @param der: a positive integer - the highest order of polynomial terms to be generated
    @return: a DataFrame() containing data, as well as polynomial terms of that data, up to order der
    This function really just *wraps* itertools.combinations_with_replacement()
    """
    poly_data = data.copy()
    for o in range(1, der + 1):
        for tpl in itertools.combinations_with_replacement(data.columns, o):
            name = "_x_".join(tpl)
            poly_data[name] = data[list(tpl)].prod(axis=1)
    return poly_data
