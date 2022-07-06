import pandas as pd

if __name__=="__main__":
    data=pd.read_pickle('save/cluster.pkl')
    print(data)