import pandas as pd
import numpy as np
from typing import Tuple, Optional
from sklearn.preprocessing import TargetEncoder, OneHotEncoder

def target_encode(
    df: pd.DataFrame, 
    column_to_encode: str, 
    target_column: str, 
    encoder: Optional[TargetEncoder] = None
) -> Tuple[pd.DataFrame, TargetEncoder]:
    """
    Encodes a categorical column using Scikit-Learn's TargetEncoder.
    
    Args:
        df: The input DataFrame.
        column_to_encode: The name of the categorical column to encode.
        target_column: The name of the target column.
        encoder: An optional pre-fitted TargetEncoder instance.
        
    Returns:
        A copy of the DataFrame with the encoded column, and the fitted TargetEncoder.
    """
    df_copy = df.copy()
    
    # TargetEncoder expects a 2D array for X
    X = df_copy[[column_to_encode]]
    y = df_copy[target_column]
    
    if encoder is None:
        # Use 'smooth' to handle rare categories. 
        # Categories not seen during fit are encoded with the target mean.
        # target_type='continuous' ensures it treats the target as a continuous variable.
        encoder = TargetEncoder(smooth='auto', target_type='continuous')
        encoder.fit(X, y)
    
    # Transform the column
    # TargetEncoder.transform returns a numpy array, we assign it back
    # We flatten because transform returns a 2D array (n_samples, 1)
    df_copy[f"{column_to_encode}_encoded"] = encoder.transform(X).flatten()
    
    return df_copy, encoder

def one_hot_encode(
    df: pd.DataFrame, 
    column_to_encode: str, 
    encoder: Optional[OneHotEncoder] = None
) -> Tuple[pd.DataFrame, OneHotEncoder]:
    """
    Encodes a categorical column using Scikit-Learn's OneHotEncoder.
    
    Args:
        df: The input DataFrame.
        column_to_encode: The name of the categorical column to encode.
        encoder: An optional pre-fitted OneHotEncoder instance.
        
    Returns:
        A copy of the DataFrame with the one-hot encoded columns, and the fitted OneHotEncoder.
    """
    df_copy = df.copy()
    
    X = df_copy[[column_to_encode]]
    
    if encoder is None:
        # sparse_output=False ensures we get a dense array back
        # handle_unknown='ignore' handles categories not seen during fit
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoder.fit(X)
    
    # Transform the column
    encoded_array = encoder.transform(X)
    
    # Get feature names
    new_columns = encoder.get_feature_names_out([column_to_encode])
    
    # Create a temporary DataFrame for the encoded features
    encoded_df = pd.DataFrame(
        np.array(encoded_array), 
        columns=new_columns.tolist(), 
        index=df_copy.index
    )

    
    # Join the new columns to the copy
    df_copy = pd.concat([df_copy, encoded_df], axis=1)
    
    return df_copy, encoder

