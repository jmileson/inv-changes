import numpy as np
import pandas as pd
from os import getcwd
from os.path import join, dirname

DEFAULT_EXPIRED = np.datetime64('2079-06-06')
DATA_GEN_DATE = np.datetime64('2017-10-05T00:00:00.000')
DATA_DIR = join(dirname(getcwd()), 'data')
DATA_FILE = join(DATA_DIR, 'WebsiteUploads_dbo_ProductInventory.csv')
HDF_DATA = join(DATA_DIR, 'data.h5')
KEY = 'inventory'
HEADERS = ['sku', 'bulk_id', 'size_code', 'quantity_on_hand', 'lifecycle_status_flag', 'expected_date',
           'location', 'back_order_type', 'product_code', 'product_type', 'is_perishable', 'is_dropship',
           'street_address_required', 'effective_on', 'expired_on', 'added_on', 'added_by', 'modified_on',
           'modified_by', 'version']
USE_COLS = ['sku', 'bulk_id', 'size_code', 'quantity_on_hand', 'lifecycle_status_flag', 'expected_date',
            'effective_on', 'expired_on']
DATE_COLS = ['expected_date', 'effective_on', 'expired_on']

COLUMN_TYPES = {
    'sku': str,
    'bulk_id': str,
    'size_code': str,
    'lifecycle_status_flag': str
}

OUT_OF_STOCK_FLAGS = ['DS', 'ED']
BACKORDERED_FLAGS = ['DX', 'EX']
STOCK_STATUS_LABELS = {
    0: 'out_of_stock',
    1: 'back_ordered',
    2: 'in_stock'
}


def apply_stock_status(df):
    data = df.copy()
    data.loc[data['lifecycle_status_flag'].isin(OUT_OF_STOCK_FLAGS), 'stock_status'] = 0  # out of stock
    data.loc[data['lifecycle_status_flag'].isin(BACKORDERED_FLAGS) & data['expected_date'], 'stock_status'] = 1  # backordered
    data['stock_status'] = data['stock_status'].fillna(2)  # in stock
    return data


def remove_rows(df):
    data = df.copy()
    data = data[~data['sku'].isnull()]
    data = data[~data['size_code'].isin(['99', '98', '97'])]
    data = data[~data['sku'].str.contains('9SPL')]
    data = data[~data['sku'].str.contains('SPCL')]

    # remove skus that are always out of stock
    status_counts = data[['sku', 'stock_status']].pivot_table(index='sku',
                                                            columns=['stock_status'],
                                                            values='stock_status',
                                                            aggfunc=lambda x: x.count()).fillna(0)
    for col in STOCK_STATUS_LABELS.keys():
        if col not in status_counts:
            status_counts[col] = 0
    always_out_of_stock = status_counts[(status_counts[1] == 0) & (status_counts[2] == 0)].index.values
    data = data[~data['sku'].astype(str).isin(always_out_of_stock.astype(str))]
    return data


def fix_ranges(df):
    # rows that have effective_on_date != expired_on_date need to apply to both the
    # end of the day they were effective on, and the beginning of the day they expire
    # on. add new rows for the 'expired on day' pull forward some information
    data = df.copy()
    span_row_filter = data['effective_on_date'] != data['expired_on_date']
    add_rows = data[span_row_filter]
    data.loc[span_row_filter, 'expired_on_date'] = data[span_row_filter]['effective_on_date'] + np.timedelta64(1, 'D')
    data.loc[span_row_filter, 'expired_on'] = data[span_row_filter]['expired_on_date']
    add_rows = add_rows.assign(effective_on_date=add_rows['expired_on_date'])
    add_rows = add_rows.assign(effective_on=add_rows['expired_on_date'])
    return data.append(add_rows, ignore_index=True)


def write_to_hdf(df, path=HDF_DATA, key=KEY, append=False):
    # tables doesn't like cols that mix np.NaN and str typess
    out_df = df.copy()
    str_cols = ['sku', 'bulk_id', 'size_code', 'lifecycle_status_flag']
    for col in str_cols:
        out_df[col] = out_df[col].astype(str)
    if append:
        mode = 'a'
    else:
        mode = 'w'
    out_df.to_hdf(path, key, format='table', mode=mode)


def load_from_csv(path=DATA_FILE):
    df = pd.read_csv(DATA_FILE,
                     names=HEADERS,
                     header=None,
                     usecols=USE_COLS,
                     parse_dates=DATE_COLS,
                     dtype=COLUMN_TYPES)

    # facilitate grouping by day
    df['effective_on_date'] = df['effective_on'].values.astype('datetime64[D]')
    df['expired_on_date'] = df['expired_on'].values.astype('datetime64[D]')

    # apply stock status
    df = apply_stock_status(df)

    # remove skus we don't care about
    df = remove_rows(df)

    # preparation
    # fix_ranges(df)

    # fix default expiration dates to day data was created
    df.loc[df['expired_on'] == DEFAULT_EXPIRED] = DATA_GEN_DATE

    # calculate the status duration
    df['status_duration'] = df['expired_on'] - df['effective_on']

    return df


def load_from_hdf(path=HDF_DATA, key=KEY):
    return pd.read_hdf(path, key)
