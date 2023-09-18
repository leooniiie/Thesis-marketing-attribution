#%% md
# This notebook is here for preprocessing the data
#%%
import pandas as pd
import numpy as np

from numpy import random
#from sklearn.preprocessing import OrdinalEncoder


def prep_data():
    random.seed(246)
    # df = pd.read_csv("data_sample1.csv")
    df_train = pd.read_csv("smaller_train.csv")
    df_valid = pd.read_csv("smaller_valid.csv")
    df_test = pd.read_csv("smaller_test.csv")

    df = pd.concat([df_train, df_test, df_valid], ignore_index=True)

    df['time_diff'] = df['timestamp_conversion'] - df['timestamp']  # create new var for timedifference
    # All observations where there's a touchpoint after conversion... 34 rows will be ignored

    df.drop(df[df.time_diff < 0].index, inplace=True)  # remove these time_diff < 0 i.e. tp after transaction

    df = df.sort_values('timestamp')
    df = df.sort_values('journey_id')

    # Long Journeys

    max_journ_len = 16
    df = df.groupby('journey_id').filter(lambda x: len(x) <= max_journ_len)

    #  Dummy variables for country, platform and channel, better but also huge data

    df = pd.get_dummies(df, columns=['channel_id'], prefix='channel', prefix_sep='_', dtype=float)
    df = pd.get_dummies(df, columns=['country_name'], prefix='country', prefix_sep='_', dtype=float)
    df = pd.get_dummies(df, columns=['platform'], prefix='platform', prefix_sep='_', dtype=float)

    # Remove irrelevant columns
    df = df.drop(['s', 'timestamp_conversion', 'time_diff'], axis=1)  # cant be used for prediction

    # Put timestamp at the last position in the data set
    df.insert(len(df.columns) - 1, 'timestamp', df.pop('timestamp'))

    return df, df.columns


def mta2tensor(df, max_journ_len):
    df_transaction = df['transaction']
    df = df.drop('transaction', axis=1)

    x = []
    y = []

    grouped = df.groupby('journey_id')
    for group_id, group_data in grouped:
        x1 = group_data.drop(['journey_id'], axis=1).values
        y_prop = df_transaction.loc[group_data.index].values

        y1 = 1 if y_prop[0] == 1 else 0
        padding = max_journ_len - len(x1)
        if padding > 0:
            x1 = np.pad(x1, ((0, padding), (0, 0)), 'constant')

        x.append(x1)
        y.append(y1)

    return x, y
