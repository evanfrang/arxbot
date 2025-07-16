import pandas as pd

def load_arxiv_data(parequet_path):
    df = pd.read_parquet(parequet_path)
    df = df.dropna()
    df = df[df['broad_category'] == 'hep_grav']
    print(df.value_counts())
    return df.to_dict(orient='records')