import pandas as pd
import numpy as np
from src.house_prices.dataloader import load_dataset
from src.house_prices.features import engineer_house_features
from src.house_prices.preprocessing import target_encode, one_hot_encode, log_transform

def inspect_data():
    path = "/home/mordicus/.cache/kagglehub/datasets/debayank2024/house-price-prediction/versions/1/"
    df = load_dataset(path)
    
    print("Original Price Stats:")
    print(df['price'].describe())
    
    df = engineer_house_features(df)
    df = log_transform(df, 'price')
    df, _ = target_encode(df, 'cityzip', 'price')
    df.drop(columns=['cityzip'], inplace=True)
    df, _, _ = one_hot_encode(df, 'street_type', drop_first=True)
    df.drop(columns=['street_type'], inplace=True)
    df = log_transform(df, 'sqft_living')
    df = log_transform(df, 'sqft_lot')
    df.drop(columns=['price_per_sqft'], inplace=True, errors='ignore')
    
    X = df.drop(columns=['price'])
    y = df['price']
    
    print("\nLog Transformed Price Stats:")
    print(y.describe())
    print("\nFeatures Summary:")
    print(X.describe().transpose())

if __name__ == "__main__":
    inspect_data()
