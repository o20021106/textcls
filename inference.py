from textcls.preprocess.preprocess_clean_seg import segmentation_chunk
from textcls.config import CNN_TOKENIZER
from argparse import ArgumentParser
import pandas as pd
import numpy as np
from textcls.preprocess.preprocess_cnn import preprocess_cnn_prediction
from textcls.preprocess.preprocess_tfidf import preprocess_tfidf_prediction
from textcls.config import MAX_SEQUENCE_LENGTH
from keras.models import load_model
import os
import pickle
import logging.config

parser = ArgumentParser()
parser.add_argument('-f', '--filename', help='optional argument', dest='filename', required=True)
parser.add_argument('-p', '--prediction_filename', help='optional argument', 
                    dest='prediction_filename', required=True)

logging.config.fileConfig(fname='textcls/base_logger.conf', disable_existing_loggers=False,
                          defaults={'logfilename': 'logs/logistic_training.log'})
logger = logging.getLogger('simpleExample')

args = parser.parse_args()
filename = args.filename
prediction_filename = args.prediction_filename

def model_predict(data):
    cnn_model = load_model('model_files/models/cnn.h5')
    with open('model_files/models/lgbm.pickle', 'rb') as f:
        lgbm_model = pickle.load(f)    
    with open('model_files/models/logistic.pickle', 'rb') as f:
        logistic_model = pickle.load(f)
    X_cnn = preprocess_cnn_prediction(data, 
                                      os.path.join('model_files/tokenizers', CNN_TOKENIZER),
                                      MAX_SEQUENCE_LENGTH)
    X_tfidf = preprocess_tfidf_prediction(data)

    cnn_prediction = cnn_model.predict(X_cnn)
    lgbm_prediction = lgbm_model.predict(X_tfidf)
    logistic_prediction = logistic_model.predict_proba(X_tfidf)
    (cnn_prediction == cnn_prediction.max(axis=1)[:,None]).astype(int)
    (lgbm_prediction == lgbm_prediction.max(axis=1)[:,None]).astype(int)
    (logistic_prediction == logistic_prediction.max(axis=1)[:,None]).astype(int)
    predictions = cnn_prediction + lgbm_prediction + logistic_prediction
    predictions = predictions.argmax(axis=1)    
    logger.info('finished model prediction')
    return predictions

if __name__ == '__main__': 
    data = pd.read_csv(f'data/input/{filename}.csv')
    data = segmentation_chunk(data)
    predictions = model_predict(data)
    data['category_int_prediction'] = predictions
    data[['title', 'content', 'category_in']].to_csv(f'data/predictions/{prediction_filename}.csv', index = False)
