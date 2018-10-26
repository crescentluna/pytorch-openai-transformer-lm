import os
import csv
import numpy as np

from tqdm import tqdm

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

seed = 3535999445


def _slots(path):
    with open(path, encoding='utf_8') as f:
        f = csv.reader(f)
        st = []
        y = []
        for i, line in enumerate(tqdm(list(f), ncols=80, leave=False)):
            if i > 0:
                s = line[0]
                st.append(s)
                y.append(int(line[-1]))
        return st, y


def slots(data_dir, n_train=1497, n_valid=200):
    storys, ys = _slots(os.path.join(data_dir, 'price_train.csv'))
    teX1, _ = _slots(os.path.join(data_dir, 'price_test.csv'))
    tr_storys, va_storys, tr_ys, va_ys = train_test_split(storys, ys, test_size=n_valid, random_state=seed)
    trX1, trX2, trX3 = [], [], []
    trY = []
    for s, y in zip(tr_storys, tr_ys):
        trX1.append(s)
        trY.append(y)

    vaX1, vaX2, vaX3 = [], [], []
    vaY = []
    for s, y in zip(va_storys, va_ys):
        vaX1.append(s)
        vaY.append(y)
    trY = np.asarray(trY, dtype=np.int32)
    vaY = np.asarray(vaY, dtype=np.int32)
    return (trX1, trY), (vaX1, vaY), (teX1, )
