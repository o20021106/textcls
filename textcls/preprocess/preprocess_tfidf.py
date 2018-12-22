#-*- coding:utf-8 -*-
import sys
import pandas as pd
import numpy as np
import scipy.sparse
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import logging.config
from textcls.config import TFIDF_VECTORIZER
import os

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
    data.dropna(inplace=True)
    logger.info('start vectorizing')
    vectorizer = TfidfVectorizer(stop_words=stopwords)
    X = vectorizer.fit_transform(data['words'])
    with open(os.path.join('model_files/tokenizers', TFIDF_VECTORIZER), 'wb') as f:
        pickle.dump(vectorizer, f)
    logger.info('finished vectorizing')
    data['category_int'] -= 1
    return X, data['category_int']


def preprocess_tfidf_prediction(data):
    with open(os.path.join('model_files/tokenizers', TFIDF_VECTORIZER), 'rb') as f:
        vectorizer = pickle.load(f)
    X = vectorizer.transform(data['words'])
    return X
