import numpy as np
import pandas as pd
import gc
import sys
import pickle
from sklearn.utils import class_weight
from textcls.preprocess.preprocess_for_training import preprocess_for_training
from textcls.models.create_model import create_model
from textcls.config import MAX_SEQUENCE_LENGTH, EMBEDDING_DIM, VALIDATION_SPLIT, TEST_SPLIT, data_paths
import logging
import logging.config
from keras.callbacks import EarlyStopping, ModelCheckpoint


logging.config.fileConfig(fname='textcls/base_logger.conf', disable_existing_loggers=False,
                          defaults={'logfilename': 'logs/cnn_training.log'})
logger = logging.getLogger('simpleExample')

data_paths = ['/root/projects/textClassification/'+path for path in data_paths] 

# preprocess and data
x_train, y_train, x_val, y_val, x_test, y_test, word_index = preprocess_for_training(data_paths,-1,
                                                                                     'model_files/tokenizers/cnn_700_words_100_dim.pickle',
                                                                                     MAX_SEQUENCE_LENGTH,
                                                                                     VALIDATION_SPLIT,
                                                                                     TEST_SPLIT)

logger.info('Finished processing CNN input data')
logger.info('y shape', y_train.shape)
# create model
model = create_model(EMBEDDING_DIM, MAX_SEQUENCE_LENGTH, 9, word_index)
# fit model
model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=10, batch_size=128, 
          callbacks=[EarlyStopping(patience=3),
          ModelCheckpoint('model_files/models/{epoch:02d}-{val_loss:.4f}.pkl',
                           save_best_only=True)])

