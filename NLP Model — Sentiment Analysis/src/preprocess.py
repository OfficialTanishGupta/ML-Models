import pandas as pd

def load_data(path):
    df = pd.read_csv(path)
    df['sentiment'] = df['sentiment'].map({'positive':1,'negative':0})
    return df