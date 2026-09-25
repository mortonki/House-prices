import numpy as np
import pandas as pd
from src.house_prices.preprocessing import log_transform, inverse_log_transform

def test_transformation():
    # Sample data
    data = pd.DataFrame({
        'price': [100000, 500000, 1000000, 2500000, 5000000]
    })
    
    print("Original Prices:")
    print(data['price'].values)
    
    # Apply log transform
    transformed_data = log_transform(data, 'price')
    print("\nTransformed (log1p):")
    print(transformed_data['price'].values)
    
    # Apply inverse transform
    reverted_data = inverse_log_transform(transformed_data, 'price')
    print("\nReverted (expm1):")
    print(reverted_data['price'].values)
    
    # Check equality
    np.testing.assert_array_almost_equal(data['price'].values, reverted_data['price'].values)
    print("\nSuccess: Transformation is mathematically correct!")

if __name__ == "__main__":
    test_transformation()
