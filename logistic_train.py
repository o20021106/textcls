# -*- coding: utf-8 -*-
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pickle
import logging.config
from textcls.preprocess.preprocess_tfidf import preprocess_tfidf
from textcls.config import data_paths

logging.config.fileConfig(fname='textcls/base_logger.conf', disable_existing_loggers=False,
                          defaults={'logfilename': 'logs/logistic_training.log'})
logger = logging.getLogger('simpleExample')

data_paths = ['/root/projects/textClassification/'+path for path in data_paths]

logger.info('preprocessing data')
X, y = preprocess_tfidf(data_paths)
y = y+1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

#create and train model
logisticRegr = LogisticRegression()
logisticRegr.fit(X_train, y_train)

logger.info('finished_training')

with open('model_files/models/model/model/logistic_model.pickle') as f:
    pickle.dump(logisticRegr, f)

