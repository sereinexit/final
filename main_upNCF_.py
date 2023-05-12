import sys
sys.path.append("/home/wyf/MF-test")
import os
import yaml
import time
import numpy as np
from models.upncf import upncf
from pathlib import Path
from scipy.sparse import load_npz

path='/home/wyf/MF-test/'
train_path = path+'data/train_user.npz'
valid_path = path+'data/valid_user.npz'
test_path = path+'data/test.npz'
table_path = path+'tables/'
opath = 'yahooR3/op_upncf_tuning_u.csv'
dataset = 'yahooR3/'

train = load_npz(train_path).tocsr()
validation = load_npz(valid_path).tocsr()
test = load_npz(test_path).tocsr()

result, topk_prediction, valid_rating_prediction, test_rating_prediction = upncf(train, validation, test, embeded_matrix=np.empty(0), iteration=100, seed=0, source=None,
                            problem='yahooR3/', gpu_on=True, scene='u', metric='AUC', topK=50, is_topK=False,
                            searcher='grid')

