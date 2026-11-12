import pandas as pd

def load_table(name):
    # reads one of the csv files from data/raw
    return pd.read_csv("data/raw/" + name + ".csv")