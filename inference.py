from textcls.preprocess.preprocess_clean_seg import segmentation_chunk
from argparse import ArgumentParser
import pandas as pd
from textcls.preprocess.preprocess_cnn import preprocess_cnn_prediction
from textcls.preprocess.preprocess_tfidf import preprocess_tfidf_prediction
from textcls.config import MAX_SEQUENCE_LENGTH
from keras.models import load_model
import os
import pickle
import logging.config

parser = ArgumentParser()
parser.add_argument('-f', '--filename', help='optional argument', dest='filename', required=True)
parser.add_argument('--cnn' , help='optional argument', dest='cnn', required=True)
parser.add_argument('--lgbm' , help='optional argument', dest='lgbm')
parser.add_argument('--logistic', help='optional argument', dest='logistic')

logging.config.fileConfig(fname='textcls/base_logger.conf', disable_existing_loggers=False,
                          defaults={'logfilename': 'logs/logistic_training.log'})
logger = logging.getLogger('simpleExample')


args = parser.parse_args()
filename = args.filename
cnn_filename = args.cnn
lgbm_filename = args.lgbm
logistic_filename = args.logistic

cnn_model = load_model(os.path.join('model_files/models', cnn_filename))
with open(os.path.join('model_files/models/', lgbm_filename), 'rb') as f:
    lgbm_model = pickle.load(f)    
with open(os.path.join('model_files/models/', logistic_filename), 'rb') as f:
    logistic_model = pickle.load(f)

data = pd.read_csv(f'data/{filename}.csv')
data = segmentation_chunk(data)

X_cnn = preprocess_cnn_prediction(data,  'model_files/tokenizers/cnn_700_words_100_dim.pickle',
                      MAX_SEQUENCE_LENGTH)
X_tfidf = preprocess_tfidf_prediction(data)

logger.info(f'cnn X: {X_cnn.shape}')
logger.info(f'tfidf X: {X_tfidf.shape}')

cnn_prediction = cnn_model.predict(X_cnn)
lgbm_prediction = lgbm_model.predict(X_tfidf)
logistic_prediction = logistic_model.predict(X_tfidf)

logger.info(f'cnn prediction shape: {cnn_prediction.shape}')
logger.info(f'lgbm prediction shape: {lgbm_prediction.shape}')
logger.info(f'logistic prediction shape: {logistic_prediction.shape}')
