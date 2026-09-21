import re

def extract_street_info(pattern, street_name):
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