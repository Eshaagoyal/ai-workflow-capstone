import os
import re
import numpy as np
import pandas as pd

def fetch_data(data_dir):
    """
    Ingest all json files from the target directory and concatenate into a single DataFrame.
    """
    if not os.path.exists(data_dir):
        raise Exception(f"Directory {data_dir} does not exist.")
        
    files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.json')]
    df_list = []
    for f in sorted(files):
        df = pd.read_json(f)
        # Standardize column naming across monthly data batches
        df.rename(columns={
            'StreamID': 'stream_id',
            'TimesViewed': 'times_viewed',
            'total_price': 'price'
        }, inplace=True)
        df_list.append(df)
        
    df_all = pd.concat(df_list, ignore_index=True)
    df_all['invoice_date'] = pd.to_datetime(df_all[['year', 'month', 'day']])
    return df_all

def engineer_features(df, training=True):
    """
    Extract lag features and rolling statistics for time-series forecasting.
    """
    df = df.sort_values(by='invoice_date').reset_index(drop=True)
    
    # 7-day and 30-day rolling sums and means for revenue
    df['revenue_7d'] = df['price'].rolling(window=7, min_periods=1).sum()
    df['revenue_30d'] = df['price'].rolling(window=30, min_periods=1).sum()
    df['views_7d'] = df['times_viewed'].rolling(window=7, min_periods=1).sum()
    
    # Target variable: next 30 days projected revenue
    if training:
        df['target'] = df['price'].rolling(window=30, min_periods=1).sum().shift(-30)
        df = df.dropna().reset_index(drop=True)
        
    return df