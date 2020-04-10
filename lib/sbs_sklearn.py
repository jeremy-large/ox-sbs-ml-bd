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


def train_n_test(Xm, y, n_folds, verbose=True):
    """
    @param Xm: rectangle of feature data
    @param y: list or array of target data
    @param n_folds: the number of splits, or folds, of the data that we would like performed
    @param verbose: Boolean. If True, report on intermediate steps
    @return : a list of floats, each is the test R2 from a fold of the data
    """

    kfold = KFold(n_splits=n_folds, shuffle=True)
    splits = kfold.split(Xm, y)

    scores = []
    for (train, test) in splits:

        if verbose or (scores == []):
            print(f"In study {len(scores) + 1}/{n_folds}, {len(test)} datapoints were held back for testing; "
                  f"first 10 such points = {test[:10]}")

        reg = LinearRegression()
        reg.fit(Xm.iloc[train], y.iloc[train])

        r2 = metrics.r2_score(  # r2_score() takes two arguments ...
            y.iloc[test],  # the actual targets, and ...
            reg.predict(Xm.iloc[test])  # the fitted targets.
        )  # We get a number between 0 and 1 back

        scores.append(r2)

    return scores


def mfe_r2_diag(data_array, histogram=False):
    """
    @param data_array: a list or array of quality data from a series of statistical fits
    @param histogram: Boolean. If True, plot a histogram, otherwise plot a scatter of all the data
    @return : nothing
    """
    n = len(data_array)
    if histogram:
        plt.hist(data_array, bins=30)
        plt.axvline(plt.mean(data_array), color='r')
    else:
        plt.scatter(range(n), data_array, marker='.')
        plt.axhline(np.mean(data_array), color='r')
    plt.title(f"R2, as calculated *only* on the testing datapoints from {n} different k-fold splits")
    plt.grid()
    plt.xlabel(f'Average value of the data, {np.round(np.mean(data_array), 3)}, is shown in red')
    plt.axhline(0, color='k')
    plt.show()
