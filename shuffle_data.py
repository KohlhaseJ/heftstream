import sys
import pandas as pd
from os.path import splitext

if __name__ == '__main__':
    arg = sys.argv[1]
    new_file = f'{splitext(arg)[0]}_shuffled.csv'
    print(f'Shuffling file {arg} to {new_file}')
    df = pd.read_csv(arg, header=None)
    ds = df.sample(frac=1)

    ds.to_csv(new_file)