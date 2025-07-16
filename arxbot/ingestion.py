import pandas as pd

def load_arxiv_data(parequet_path, parquet_out):
    df = pd.read_parquet(parequet_path)
    df = df.dropna()
    df = df[df['broad_category'] == 'hep_grav']
    print(df.value_counts())
    df.to_parquet(parquet_out)
    return df.to_dict(orient='records')