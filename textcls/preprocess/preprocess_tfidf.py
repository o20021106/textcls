#-*- coding:utf-8 -*-
import sys
import pandas as pd
import numpy as np
import scipy.sparse
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import logging.config

logging.config.fileConfig(fname='textcls/base_logger.conf', disable_existing_loggers=False,
                          defaults={'logfilename': 'logs/lgbm_training.log'})
logger = logging.getLogger('simpleExample')


def preprocess_tfidf(data_paths):
    stopwords  = []
    with open('utils/stopwords.txt', 'r') as f:
        lines = f.readlines()
        stopwords = [word.strip() for word in lines]
    data = []
    for path in data_paths:
        logger.debug(path)
        temp =  pd.read_csv(path, lineterminator = '\n')
        data.append(temp[['category_int','words']])
    data = pd.concat(data)
    data_without_health_sports = data[data['category_int']!=3]
    data_health_sports = data[data['category_int']==3]
    #data_health_sports = data_health_sports.sample(70000)
    data = pd.concat([data_without_health_sports, data_health_sports])
    data.dropna(inplace=True)
    logger.info('start vectorizing')
    vectorizer = TfidfVectorizer(stop_words=stopwords)
    X = vectorizer.fit_transform(data['words'])
    logger.info('finished vectorizing')
    data['category_int'] -= 1
    return X, data['category_int']
