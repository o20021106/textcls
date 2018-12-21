# -*- coding: utf-8 -*-
import pickle
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import logging.config
from textcls.preprocess.preprocess_tfidf import preprocess_tfidf
from textcls.config import data_paths

logging.config.fileConfig(fname='textcls/base_logger.conf', disable_existing_loggers=False,
                          defaults={'logfilename': 'logs/lgbm_training.log'})
logger = logging.getLogger('simpleExample')

data_paths = ['/root/projects/textClassification/'+path for path in data_paths]

logger.info('preprocessing data')
X, y = preprocess_tfidf(data_paths)
#X = scipy.sparse.load_npz('data/tfidf/'+sys.argv[1]+'.npz')
#data = pd.read_csv('data/tfidf/'+sys.argv[2]+'.csv')
#data['category_int'] = data['category_int']-1
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.1, random_state=42)
train_data = lgb.Dataset(X_train, label=y_train)
valid_data = lgb.Dataset(X_test, label=y_test)

# create and train model
param = {'objective': 'multiclass', 'num_class':9}
param['metric'] = ['multi_logloss']
num_round = 400

lgbm = lgb.train(param, train_data, num_round, valid_sets=valid_data, verbose_eval=10)

logger.info('finished training')

# save model
with open('model_files/models/lgbm.pickle', 'wb') as f:
    pickle.dump(lgbm, f)

