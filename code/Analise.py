import pandas as pd


if __name__ == '__main__':
    result = pd.read_pickle('../metrics/metrics_summary.pkl')
    abc = result.describe()
    print result.head()
