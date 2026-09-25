import pandas as pd
import numpy as np
import re
from typing import Tuple, Optional

def extract_street_info(pattern: str, street_name: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extracts street name and type from a street address string.
    """
    if not isinstance(street_name, str):
        return None, None
    
    match = re.search(pattern, street_name, re.IGNORECASE)
    if match:
        suffix = match.group(1)
        # The name is everything before the suffix
        name = street_name.split(suffix)[0].strip()
        return name, suffix
    else:
        # If no suffix found, return the whole string as name and 'Other' as suffix
        return street_name, 'Other'

def engineer_house_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs feature engineering as identified in the EDA notebook.
    """
    df = df.copy()
    
    # Date Features
    df['date'] = pd.to_datetime(df['date'])
    df['weekday'] = df['date'].dt.weekday
    df['days_since_first'] = (df['date'] - df['date'].min()).dt.days
    df.drop(columns=['date'], inplace=True)
    
    # Renovation Features
    df.loc[df['yr_renovated'] <= 0, 'yr_renovated'] = df['yr_built']
    df['was_renovated'] = (df['yr_renovated'] > df['yr_built']).astype(int)
    
    # Basement Features
    df['has_basement'] = (df['sqft_basement'] > 0).astype(int)
    df.drop(columns=['sqft_above', 'sqft_basement'], inplace=True)
    
    # Street Information
    suffixes = [
        'Ave', 'Avenue', 'St', 'Street', 'Blvd', 'Boulevard', 
        'Dr', 'Drive', 'Ln', 'Lane', 'Ct', 'Court', 'Way', 
        'Ter', 'Terrace', 'Cir', 'Circle', 'Pl', 'Place', 
        'Rd', 'Road', 'Parkway', 'Trace', 'Trail'
    ]
    suffixes.sort(key=len, reverse=True)
    pattern = r'\b(' + '|'.join(map(re.escape, suffixes)) + r')\b'
    
    df[['street_name', 'street_type']] = df['street'].apply(
        lambda x: pd.Series(extract_street_info(pattern=pattern, street_name=x))
    )
    
    suffix_mapping = {
        'Ave': 'Avenue', 'Avenue': 'Avenue',
        'St': 'Street', 'Street': 'Street',
        'Blvd': 'Boulevard', 'Boulevard': 'Boulevard',
        'Dr': 'Drive', 'Drive': 'Drive',
        'Ln': 'Lane', 'Lane': 'Lane',
        'Ct': 'Court', 'Court': 'Court',
        'Rd': 'Road', 'Road': 'Road',
        'Pl': 'Place', 'Place': 'Place',
        'Cir': 'Circle', 'Circle': 'Circle',
        'Ter': 'Terrace', 'Terrace': 'Terrace',
        'Way': 'Way', 'Tr': 'Trace', 'Trail': 'Trail'
    }
    
    df['street_type'] = df['street_type'].map(suffix_mapping).fillna(df['street_type'])
    
    threshold = 0.02 * len(df)
    counts = df['street_type'].value_counts()
    rare_types = counts[counts < threshold].index
    df['street_type'] = df['street_type'].apply(lambda x: 'Other' if x in rare_types else x)
    
    # Geographic Features
    df[['state', 'zip']] = df['statezip'].str.split(' ', expand=True)
    df.drop(columns=['statezip'], inplace=True)
    df.drop(columns=['state'], inplace=True)
    df['cityzip'] = df['city'] + "_" + df['zip']
    df.drop(columns=['city', 'zip'], inplace=True)
    
    # Final cleanup
    df.drop(columns=['street', 'street_name'], inplace=True)
    
    return df
