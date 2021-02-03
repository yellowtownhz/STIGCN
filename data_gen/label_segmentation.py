#! /usr/bin/env python
import numpy as np
import pickle
import json

total = 400
split = 350
a = np.random.permutation(total)
train = a[:split]
test = a[split:]
train = [int(x) for x in train]
test = [int(x) for x in test]
train.sort()
test.sort()
print(f"train: {train}")
print(f"test: {test}")
ret = {'train': train, 'test': test}
with open('../data/kinetics_raw/label_segmentation.json', 'w') as f:
    json.dump(ret, f)

# with open('../data/kinetics_raw/label_segmentation.pkl', 'wb') as f:
    # pickle.dump((train, test), f)

