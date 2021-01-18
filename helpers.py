import numpy as np
import pandas as pd
from dirty_cat import SimilarityEncoder
from pathlib import Path
import holidays


def loguniform(low=0, high=1, size=None, base=10):
    """Returns a number or a set of numbers from a log uniform distribution"""
    return np.power(base, np.random.uniform(low, high, size))


def encode_dates(df, column, country=None):
    df.copy()
    if country:
        country_holidays = holidays.CountryHoliday(country)
        df["holiday"] = df[column].isin(country_holidays)
    df[column + "_year"] = df[column].dt.year
    df[column + "_month"] = df[column].dt.month
    df[column + "_day"] = df[column].dt.day

    df[column + "_hour"] = df[column].dt.hour
    df[column + "_minute"] = df[column].dt.minute
    df[column + "_second"] = df[column].dt.second
    df = df.drop(column, axis=1)
    return df


def similarity_encode(X, encode_columns, n_prototypes, train, drop_original):
    X = X.copy()
    if train:
        enc = SimilarityEncoder(
            similarity="ngram", categories="k-means", n_prototypes=n_prototypes
        )
        enc.fit(X[encode_columns].values)
        Path("encoders").mkdir(exist_ok=True)
        pd.to_pickle(enc, "encoders/similarity_encoder.pickle")
    else:
        enc = pd.read_pickle("encoders/similarity_encoder.pickle")
    transformed_values = enc.transform(X[encode_columns].values)

    transformed_values = pd.DataFrame(transformed_values, index=X.index)
    transformed_columns = []
    for col in encode_columns:
        for i in range(0, n_prototypes):
            transformed_columns.append(col + "_" + str(i))
    transformed_values.columns = transformed_columns
    X = pd.concat([X, transformed_values], axis=1)
    if drop_original:
        X = X.drop(encode_columns, axis=1)
    return X
