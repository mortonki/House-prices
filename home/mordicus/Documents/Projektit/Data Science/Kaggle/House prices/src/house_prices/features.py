    if rare_types is None:
        threshold = 0.02 * len(df)
        counts = df['street_type'].value_counts()
        rare_types = counts[counts.lt(threshold)].index.tolist()

    df['street_type'] = df['street_type'].apply(lambda x: 'Other' if x in rare_types else x)