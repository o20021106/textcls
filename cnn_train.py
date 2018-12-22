from textcls.preprocess.preprocess_cnn import preprocess_cnn
from textcls.models.create_model import create_model
from textcls.config import MAX_SEQUENCE_LENGTH, EMBEDDING_DIM, VALIDATION_SPLIT, TEST_SPLIT, CNN_TOKENIZER
import logging.config
from keras.callbacks import EarlyStopping, ModelCheckpoint
import os

logging.config.fileConfig(fname='textcls/base_logger.conf', disable_existing_loggers=False,
                          defaults={'logfilename': 'logs/cnn_training.log'})
logger = logging.getLogger('simpleExample')

data_paths = os.listdir('data/preprocessed')
data_paths = [os.path.join('data/preprocessed', path) for path in data_paths]

# preprocess and data
(x_train, y_train, x_val, y_val,
 x_test, y_test, word_index) = preprocess_cnn(data_paths, -1,
                                              os.path.join('model_files/tokenizers', CNN_TOKENIZER),
                                              MAX_SEQUENCE_LENGTH,
                                              VALIDATION_SPLIT,
                                              TEST_SPLIT)

logger.info('Finished processing CNN input data')
# create model
model = create_model(EMBEDDING_DIM, MAX_SEQUENCE_LENGTH, 9, word_index)
# fit model
model.fit(x_train, y_train, epochs=10, batch_size=128, validation_data=(x_val, y_val),
          callbacks=[EarlyStopping(patience=3),
          ModelCheckpoint('model_files/models/cnn.h5','val_loss', save_best_only=True)])


