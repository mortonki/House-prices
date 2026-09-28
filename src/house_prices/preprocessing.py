import pandas as pd
import numpy as np
from typing import Tuple, Optional, Union, List
from sklearn.preprocessing import TargetEncoder, OneHotEncoder, StandardScaler

def target_encode(
    X: pd.DataFrame,
    column_to_encode: str,
    y: Union[str, pd.Series],
    encoder: Optional[TargetEncoder] = None
) -> Tuple[pd.DataFrame, TargetEncoder]:
    """
    Encodes a categorical column using Scikit-Learn's TargetEncoder.
    
    Args:
        X: The input DataFrame.
        column_to_encode: The name of the categorical column to encode.
        y: The name of the target column (str) or the target series (pd.Series).
        encoder: An optional pre-fitted TargetEncoder instance.
    
    Returns:
        A copy of the DataFrame with the encoded column, and the fitted TargetEncoder.
    """
    df_copy = X.copy()
    
    # Extract target
    if isinstance(y, str):
        # This case is for backward compatibility where y is a column name in df_copy
        # But since we split X and y, this won't happen in the new pipeline.
        # However, we keep it for tests.
        y_vals = df_copy[y]
    else:
        y_vals = y
        
    # TargetEncoder expects a 2D array for X
    X_subset = df_copy[[column_to_encode]]
    
    if encoder is None:
        # Use 'smooth' to handle rare categories.
        # Categories not seen during fit are encoded with the target mean.
        # target_type='continuous' ensures it treats the target as a continuous variable.
        encoder = TargetEncoder(smooth='auto', target_type='continuous')
        encoder.fit(X_subset, y_vals)
    
    # Transform the column
    # TargetEncoder.transform returns a numpy array, we assign it back
    # We flatten because transform returns a 2D array (n_samples, 1)
    df_copy[f"{column_to_encode}_encoded"] = encoder.transform(X_subset).flatten()
    
    return df_copy, encoder

def one_hot_encode(
    df: pd.DataFrame,
    column_to_encode: str,
    encoder: Optional[OneHotEncoder] = None,
    drop_first: bool = False
) -> Tuple[pd.DataFrame, OneHotEncoder, Optional[str]]:
    """
    Encodes a categorical column using Scikit-Learn's OneHotEncoder.
    
    Args:
        df: The input DataFrame.
        column_to_encode: The name of the categorical column to encode.
        encoder: An optional pre-fitted OneHotEncoder instance.
        drop_first: If True, drops the first category to avoid the dummy variable trap.
        
    Returns:
        A tuple containing:
        - A copy of the DataFrame with the one-hot encoded columns.
        - The fitted OneHotEncoder instance.
        - The name of the dropped column (None if drop_first is False).
    """
    df_copy = df.copy()
    
    X = df_copy[[column_to_encode]]
    
    if encoder is None:
        # sparse_output=False ensures we get a dense array back
        # handle_unknown='ignore' handles categories not seen during fit
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoder.fit(X)
    
    # Transform the column
    encoded_array = np.asarray(encoder.transform(X))
    
    # Get feature names
    new_columns = encoder.get_feature_names_out([column_to_encode]).tolist()
    
    dropped_col = None
    if drop_first:
        dropped_col = new_columns[0]
        # Slice the array to remove the first column
        encoded_array = encoded_array[:, 1:]
        # Update the list of column names
        new_columns = new_columns[1:]
    
    # Create a temporary DataFrame for the encoded features
    encoded_df = pd.DataFrame(
        encoded_array,
        columns=new_columns,
        index=df_copy.index
    )
    
    # Join the new columns to the copy
    df_copy = pd.concat([df_copy, encoded_df], axis=1)
    
    return df_copy, encoder, dropped_col

def log_transform(
    df: pd.DataFrame,
    column_name: str
) -> pd.DataFrame:
    """
    Applies a log1p transformation to a specified column in a DataFrame.
    Log1p is used to handle zero values gracefully (log(1+x)).
    
    Args:
        df: The input DataFrame.
        column_name: The name of the column to transform.
    
    Returns:
        A copy of the DataFrame with the transformed column.
    """
    df_copy = df.copy()
    df_copy[column_name] = np.log1p(df_copy[column_name])
    return df_copy

def inverse_log_transform(
    df: pd.DataFrame,
    column_name: str
) -> pd.DataFrame:
    """
    Applies an inverse log1p transformation (expm1) to a specified column in a DataFrame.
    
    Args:
        df: The input DataFrame.
        column_name: The name of the column to transform.
    
    Returns:
        A copy of the DataFrame with the inverse transformed column.
    """
    df_copy = df.copy()
    df_copy[column_name] = np.expm1(df_copy[column_name])
    return df_copy


def standardize(
    df: pd.DataFrame,
    columns: List[str],
    scaler: Optional[StandardScaler] = None
) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Standardizes specified columns in a DataFrame using StandardScaler.
    
    Args:
        df: The input DataFrame.
        columns: List of column names to standardize.
        scaler: An optional pre-fitted StandardScaler instance.
    
    Returns:
        A tuple containing:
        - A copy of the DataFrame with standardized columns.
        - The fitted StandardScaler instance.
    """
    df_copy = df.copy()
    
    if scaler is None:
        scaler = StandardScaler()
        scaler.fit(df_copy[columns])
    
    df_copy[columns] = scaler.transform(df_copy[columns])
    
    return df_copy, scaler
