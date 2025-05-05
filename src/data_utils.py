def load_dataset(file_path):
    import pandas as pd
    return pd.read_csv(file_path)

def save_dataset(dataframe, file_path):
    dataframe.to_csv(file_path, index=False)

def get_column_names(dataframe):
    return dataframe.columns.tolist()