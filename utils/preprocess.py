import pandas as pd

def preprocess_data(df):
    df.fillna(method='ffill', inplace=True)
    return df[['valuenum_scaled']]
