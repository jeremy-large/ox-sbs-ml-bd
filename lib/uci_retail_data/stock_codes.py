import numpy as np
import pandas as pd


def stock_code_to_num(x, ndigits=5, ignorestring='DCGS'):
    """
    :param x: a stock code
    :param ndigits: the number of digits we might expect at the start of the code
    :param ignorestring: string to ignore within the code
    :return: an integer
    """
    if type(x) == str:
        try:
            return int(x.strip(ignorestring)[:ndigits])
        except ValueError:
            print(x)
            return 0
    return x


def is_invalid(datapoint):
    """
    :param datapoint: a datapoint from the UCI opensource data
    we weed out negative quantities and zero prices.
    There are a bunch of duff Stock Codes corresponding to postage, etc.
    There are also a small number of idiosyncratic cases: 'DCGS0066N', 'DCGSSBOY', 'DCGSSGIRL', 'SP1002'
    Sometimes, the description contains information indicating that this is not describing a real purchase
    :return: whether this is a valid datapoint to process into the raw files
    """

    if datapoint['Quantity'] < 0:
        return True

    if datapoint['Price'] == 0:
        return True

    for s in ('POST', 'DOT', 'gift', 'TEST', 'BANK CHARGES', 'ADJUST',
              'PADS', 'AMAZONFEE', 'DCGS0066N', 'DCGSSBOY', 'DCGSSGIRL', 'SP1002'):
        if s in str(datapoint['StockCode']):
            return True

    if datapoint['Description'] in ('Discount', 'Manual', 'SAMPLES', 'Adjust bad debt', 'update'):
        return True

    return False


def customer_code(datapoint):
    """
    :param datapoint: row of a pandas dataframe containing the UCI retail dataset
    :return: customer code as integer
    """
    c = datapoint['Customer ID']
    if type(c) == float and np.isnan(c):
        return 0
    return int(c)


def invalid_series(datf):
    """
    :param datf:
    :return: boolean Series saying whether each item in the dataframe is_invalid()
    """
    return pd.Series([is_invalid(row) for index, row in datf.iterrows()])


def invoice_df(df, invalid_series=None):
    """
    :param df: original dataframe from UCI
    :param invalid_series: boolean Series saying whether each item in the dataframe is_invalid()
    :return: a dataframe with one row per invoice
    """
    df = df.copy()
    if invalid_series is not None:
        df = df.loc[~invalid_series]
    df['Cost'] = df.Price * df.Quantity
    df['Hour'] = df.InvoiceDate.dt.hour
    df['Month'] = df.InvoiceDate.dt.month
    df['Year'] = df.InvoiceDate.dt.year

    gb = df.groupby('Invoice')

    def words(ts):
        s = ts.apply(str).str.cat(sep=' ', na_rep='-')
        return set(s.strip().split(' ')) - {'', ' ', '  ', ' ,'}

    invoices = pd.concat([gb['Customer ID'].max(),
                          gb.StockCode.count(),
                          gb.Quantity.sum(),
                          gb.Cost.sum(),
                          gb.Hour.max(),
                          gb.Month.max() + gb.Year.max() * 100,
                          gb.Description.apply(words),
                          gb.Country.max()],
                         axis=1)

    invoices.columns = (['customer', 'n_codes', 'n_items', 'spend', 'hour', 'month', 'words', 'country'])

    invoices['words_per_item'] = invoices.words.apply(len) / invoices.n_codes

    return invoices
