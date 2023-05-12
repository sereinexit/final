import sys
sys.path.append("/home/wyf/MF-test")
import os
import yaml
import time
import numpy as np
from models.upmf1 import upmf1
from pathlib import Path
from scipy.sparse import load_npz
path='/home/wyf/MF-test/'
train_path = path+'data/train_user.npz'
valid_path = path+'data/valid_user.npz'
test_path = path+'data/test.npz'
table_path = path+'tables/'
opath = 'yahooR3/op_dib_mf1_tuning_u.csv'
dataset = 'yahooR3/'

train = load_npz(train_path).tocsr()
validation = load_npz(valid_path).tocsr()
test = load_npz(test_path).tocsr()

trials, best_params = upmf1(train, validation, test, embeded_matrix=np.empty(0), iteration=500, seed=0, source=None,
                            problem='yahooR3/', gpu_on=True, scene='u', metric='AUC', topK=50, is_topK=False,
                            searcher='optuna')

if not os.path.exists(table_path + opath):
    if not os.path.exists(table_path + dataset):
        os.makedirs(table_path + dataset)

trials.to_csv(table_path + opath)

if Path(table_path + dataset + 'op_hyper_params_u.yml').exists():
    pass
else:
    yaml.dump(dict(yahooR3=dict()),
              open(table_path + dataset + 'op_hyper_params_u.yml', 'w'), default_flow_style=False)
time.sleep(0.5)
hyper_params_dict = yaml.safe_load(open(table_path + dataset + 'op_hyper_params_u.yml', 'r'))
hyper_params_dict['yahooR3']['UPMF1'] = best_params
yaml.dump(hyper_params_dict, open(table_path + dataset + 'op_hyper_params_u.yml', 'w'),
          default_flow_style=False)


