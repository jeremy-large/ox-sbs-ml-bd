import os

import numpy as np
import pandas as pd
import logging


def stock_code_to_num(x, ndigits=5, ignorestring='DCGS'):
    """
    :param x: a stock code
    :param ndigits: the number of digits we might expect at the start of the code
    :param ignorestring: string to ignore within the code
    :return: an integer
    """
    s = str(x)
    if s.startswith(ignorestring):
        s = s[len(ignorestring):]
    try:
        five_digits = s[:ndigits]
        last_bit = "".join(str(ord(c)) for c in s[ndigits:])
        return int(last_bit + five_digits)
    except ValueError:
        logging.debug(x)
        return 0


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
    repo_dir = os.path.dirname(os.getcwd())
    local_data_file = os.path.join(repo_dir, 'data', 'invalids.csv')
    if os.path.exists(local_data_file):
        in_data = pd.read_csv(local_data_file)
        return in_data[in_data.columns[-1]]
    df = pd.Series([is_invalid(row) for index, row in datf.iterrows()])
    df.to_csv(local_data_file)
    logging.info('Saving a copy to ' + local_data_file)
    return df


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

    invoices.columns = (['customer',
                         'codes_in_invoice',
                         'items_in_invoice',
                         'invoice_spend',
                         'hour', 'month', 'words', 'country'])

    invoices['words_per_item'] = invoices.words.apply(len) / invoices.codes_in_invoice

    return invoices


def stockcode_df(df, invalid_series=None):
    """
    :param df: original dataframe from UCI
    :param invalid_series: boolean Series saying whether each item in the dataframe is_invalid()
    :return: a dataframe with one row per stock code
    """
    df = df.copy()
    df['Cost'] = df.Price * df.Quantity
    if invalid_series is not None:
        df = df.loc[~invalid_series]

    gb = df.groupby('StockCode')

    stockcodes = pd.concat([gb['Customer ID'].nunique(),
                            gb.Invoice.nunique(),
                            gb.Quantity.sum(),
                            gb.Description.apply(lambda x: x.iloc[0]),
                            gb.Price.max(),
                            gb.Price.min(),
                            gb.Cost.sum() / gb.Quantity.sum(),
                            gb.Price.std()
                            ],
                           axis=1)

    stockcodes.columns = (['customers_buying_this_code',
                           'invoices_with_this_code',
                           'n_units_sold',
                           'description',
                           'max_price',
                           'min_price',
                           'mean_price',
                           'stdev_price'])

    return stockcodes


def thin_df(df, max_stock_codes, min_customers, invalid_series):
    """
    :param df: dataframe of the open source UCI dataset
    :param max_stock_codes: if an invoice has more than this many separate stock codes, exclude it
    :param min_customers: if a stock code has fewer than this many customers in the data, exclude it
    :param invalid_series: boolean Series saying whether each item in the dataframe is_invalid()
    :return:
    """
    inv = invoice_df(df, invalid_series=invalid_series)
    stockcodes = stockcode_df(df, invalid_series=invalid_series)

    df_inv = pd.merge(df[~invalid_series], inv, left_on='Invoice', right_index=True)
    df_is = pd.merge(df_inv, stockcodes, left_on='StockCode', right_index=True)

    df_thin = df_is.copy()

    if max_stock_codes > 0:
        df_thin = df_thin[df_thin['codes_in_invoice'] < max_stock_codes]
        logging.info(f"Removed invoices with more than {max_stock_codes} different stock codes.")
        logging.info(f"{len(df_bp) - len(df_thin)} rows removed from a dataset of {len(df_bp)} rows")

    if min_customers > 0:
        orig_len = len(df_thin)
        df_thin = df_thin[df_thin['customers_buying_this_code'] > min_customers]
        logging.info(f"Removed stock codes with fewer than {min_customers} different purchasers.")
        logging.info(f"{orig_len - len(df_thin)} rows removed from a dataset of {orig_len} rows")

    df_thin.sort_index(inplace=True)

    return df_thin[df.columns]
