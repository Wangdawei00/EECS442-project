import pandas as pd

def split_total():
    df = pd.read_csv('nyu_csv/nyu2_train.csv', header=None)
    ds_total = df.sample(frac=1)
    ds_medium = df.sample(frac=0.2)
    ds_small = df.sample(frac=0.04)
    ds_total.to_csv('nyu_csv/nyu2_t.csv',index=False,header=False)
    ds_medium.to_csv('nyu_csv/nyu2_m.csv',index=False,header=False)
    ds_medium.to_csv('nyu_csv/nyu2_s.csv',index=False,header=False)

def split_trvatr():
    types = ['t', 'm', 's']
    for t in types:
        in_csv = f'nyu_csv/nyu2_{t}.csv'
        total_lines = sum(1 for row in (open(in_csv)))
        train_lines = int(total_lines*0.99*0.8)
        valid_lines = int(total_lines*0.99*0.2)
        test_lines = total_lines-train_lines-valid_lines

        df_tr = pd.read_csv(in_csv, header=None, nrows = train_lines, skiprows = 0)
        df_va = pd.read_csv(in_csv, header=None, nrows = valid_lines, skiprows = train_lines)
        df_te = pd.read_csv(in_csv, header=None, nrows = test_lines, skiprows = train_lines + valid_lines)
        df_tr.to_csv(f'nyu_csv/nyu2_train_{t}.csv',index=False,header=False)
        df_va.to_csv(f'nyu_csv/nyu2_valid_{t}.csv',index=False,header=False)
        df_te.to_csv(f'nyu_csv/nyu2_test_{t}.csv',index=False,header=False)


split_total()
split_trvatr()