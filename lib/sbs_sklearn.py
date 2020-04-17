import os
import logging
import numpy as np
import pylab as plt

from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn import metrics

from uci_retail_data import stock_codes
from uci_retail_data import uci_files


def get_standard_data():
    repo_dir = os.path.dirname(os.getcwd())
    local_data_file = os.path.join(repo_dir, 'data', 'raw.csv')

    if os.path.exists(local_data_file):
        df = uci_files.load_uci_file(local_data_file, uci_files.SHEET_NAME)
    else:
        df = uci_files.load_uci_file(uci_files.REMOTE_FILE, uci_files.SHEET_NAME)
        df.to_csv(local_data_file)
        logging.info('Saving a copy to ' + local_data_file)

    # familiar dataset:
    invalids = stock_codes.invalid_series(df)
    invoices = stock_codes.invoice_df(df, invalid_series=invalids)
    return df, invalids, invoices


def train_n_test(X, y, n_folds, update_frequency=None, model=None, metric=None, train_on_minority=False):
    """
    @param X: a pandas DataFrame of features data of shape (A, B)
    @param y: a pandas Series of target data, of length A
    @param n_folds: the number of splits, or folds, of the data that we would like performed
    @param update_frequency: after implementing this many folds, provide an update
    @param model: by default LinearRegression(). Can also be set to another model with methods .fit() and .predict()
    @param metric: by default metrics.r2_score. Can also be set to another metric
    @param train_on_minority: if set to True, then reverse the roles of test and train
    @return : a list of floats, each is the test R2 from a fold of the data
    """
    update_frequency = update_frequency or len(y) - 1
    model = LinearRegression() if model is None else model
    metric = metrics.r2_score if metric is None else metric
    y = y.values.ravel()   # strip out from y just its float values as an array
    Xm = X.values          # strip out from X just the array of data it contains

    kfold = KFold(n_splits=n_folds, shuffle=True)

    scores = []
    for (train, test) in kfold.split(Xm, y):

        if train_on_minority:
            train, test = (test, train)    # swap em!

        if len(scores) % update_frequency == 0:
            logging.info(
                f"In study {len(scores) + 1}/{n_folds}, train on {len(train)} points; then test on the other {len(test)}: "
                f"first few test points = {test[:5]} ")

        model.fit(Xm[train], y[train])

        score = metric(                              # r2_score(), and similar metrics, takes two arguments ...
                       y[test],                 # the actual targets, and ...
                       model.predict(Xm[test])  # the fitted targets.
                      )                              # We get a number (maybe between 0 and 1) back

        scores.append(score)

    return scores


def mfe_r2_diag(x, histogram=False, metric_name=None, n_bins=30):
    """
    @param x: a list or array of quality data from a series of statistical fits
    @param histogram: Boolean. If True, plot a histogram, otherwise plot a scatter of all the data
    @param metric_name: an optional string which is the name of the metric, a list of which is in x
    @return : nothing
    """

    metric_name = metric_name or 'R2'

    if histogram:
        plt.hist(x, bins=n_bins)
        plt.axvline(np.mean(x), color='r')
    else:
        plt.scatter(range(len(x)), x, marker='.')
        plt.axhline(np.mean(x), color='r')

    plt.xlabel(f'Average value of the data, {np.round(np.mean(x), 3)}, is shown in red')
    plt.title(f"{metric_name}, as calculated *only* on the testing datapoints from {len(x)} different k-fold splits")

    plt.grid(); plt.axhline(0, color='k'); plt.show()
