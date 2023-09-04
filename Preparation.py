#%% md
# This notebook is here for preprocessing the data
#%%
import pandas as pd
import numpy as np

from numpy import random
from sklearn.preprocessing import OrdinalEncoder


def prep_data():
    random.seed(246)
    df_train = pd.read_csv("smaller_train.csv")
    df_valid = pd.read_csv("smaller_valid.csv")
    df_test = pd.read_csv("smaller_test.csv")

    df = pd.concat([df_train, df_valid, df_test], ignore_index=True)

    # df_valid = pd.read_csv("data_sample1.csv")
    # df_test = pd.read_csv("data_sample2.csv")

    # df = pd.concat([df_test, df_valid], ignore_index=True)

    df['time_diff'] = df['timestamp_conversion'] - df['timestamp']  # create new var for timedifference
    # All observations where there's a touchpoint after conversion... 34 rows will be ignored

    df.drop(df[df.time_diff < 0].index, inplace=True)  # remove these time_diff < 0 i.e. tp after transaction

    df = df.sort_values('timestamp')
    df = df.sort_values('journey_id')

    # This transform df so that only last touchpoint before conversion gets transaction = 1. Because this leaves to few
    # observations with transaction == 1 we don't consider it for now

    # groups = df.groupby('journey_id').time_diff
    # min_val = groups.transform(min) #search minimal time_diff in each group <=> closest tp to conversion

    # cond1 = df.time_diff==min_val #define condition when transaction should be 1

    # df['transaction'] = np.select([cond1], [1], default = 0) #transform transaction

    # Long Journeys

    max_journ_len = 16
    df = df.groupby('journey_id').filter(lambda x: len(x) <= max_journ_len)

    #  Dummy variables for country, platform and channel, better but also huge data

    df = pd.get_dummies(df, columns=['channel_id'], prefix='channel', prefix_sep='_', dtype=float)
    df = pd.get_dummies(df, columns=['country_name'], prefix='country', prefix_sep='_', dtype=float)
    df = pd.get_dummies(df, columns=['platform'], prefix='platform', prefix_sep='_', dtype=float)

    # Ordinal Encoder, not really accurate, but doesn't blow up df

    # ordinal_encoder = OrdinalEncoder()
    # for column in df.columns:
    #    if df[column].dtypes == 'object':
    #        df[column] = ordinal_encoder.fit_transform(df[[column]])

    # Remove irrelevant columns

    df = df.drop(['s', 'timestamp_conversion', 'time_diff'], axis=1)  # cant be used for prediction

    # Put timestamp at the last position in the data set

    df.insert(len(df.columns)-1, 'timestamp', df.pop('timestamp'))

    return df, df.columns


def mta2tensor(df, max_journ_len):
    colx = df.shape[1] - 2
    df_transaction = df['transaction']
    df = df.drop('transaction', axis=1)
    grous = df.groupby('journey_id')
    x = []
    y = []
    cj_count = 1
    for i in df['journey_id'].unique():
        x1 = grous.get_group(i)
        x1 = x1.drop(['journey_id'], axis=1)
        x1 = x1.values.tolist()
        y_prop = df_transaction.loc[grous.get_group(i).index]
        y_prop = y_prop.values.tolist()
        if y_prop[0] == 1:
            y1 = 1
        else:
            y1 = 0
        for j in range(max_journ_len - len(x1)):
            x1.append([0] * colx)
        x.append(x1)
        y.append(y1)
        cj_count = cj_count + 1

    return x, y
