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


def train_n_test(Xm, y, n_folds, verbose=False, model=None):
    """
    @param Xm: rectangle of feature data
    @param y: list or array of target data
    @param n_folds: the number of splits, or folds, of the data that we would like performed
    @param verbose: Boolean. If True, report on intermediate steps
    @param model: by default None. Can also be set to another model with methods .fit() and .predict()
    @return : a list of floats, each is the test R2 from a fold of the data
    """
    model = LinearRegression() if model is None else model

    kfold = KFold(n_splits=n_folds, shuffle=True)

    scores = []
    for (train, test) in kfold.split(Xm, y):

        if verbose or (len(scores)%1000 == 0):
            print(f"In study {len(scores) + 1}/{n_folds}, {len(test)} datapoints were held back for testing; "
                  f"first few such points = {test[:10]}")

        model.fit(Xm.iloc[train], y.iloc[train])

        r2 = metrics.r2_score(                                # r2_score() takes two arguments ...
                                y.iloc[test],                 # the actual targets, and ...
                                model.predict(Xm.iloc[test])  # the fitted targets.
                             )                                # We get a number (maybe between 0 and 1) back

        scores.append(r2)

    return scores


def mfe_r2_diag(x, histogram=False, title=None, n_bins=30):
    """
    @param x: a list or array of quality data from a series of statistical fits
    @param histogram: Boolean. If True, plot a histogram, otherwise plot a scatter of all the data
    @param title: an optional string which is the title for the chart
    @return : nothing
    """

    if histogram:
        plt.hist(x, bins=n_bins)
        plt.axvline(np.mean(x), color='r')
    else:
        plt.scatter(range(len(x)), x, marker='.')
        plt.axhline(np.mean(x), color='r')

    plt.xlabel(f'Average value of the data, {np.round(np.mean(x), 3)}, is shown in red')
    plt.title(title or f"R2, as calculated *only* on the testing datapoints from {len(x)} different k-fold splits")

    plt.grid(); plt.axhline(0, color='k'); plt.show()
