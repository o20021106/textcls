# -*- coding: utf-8 -*`
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
import pickle
import logging.config
import gc

logging.config.fileConfig(fname='textcls/base_logger.conf', disable_existing_loggers=False,
                          defaults={'logfilename': 'logs/cnn_training.log'})
logger = logging.getLogger('simpleExample')


def import_data(data_paths, sample_size):
    data = []
    for path in data_paths:
        logger.info(path)
        temp = pd.read_csv(path, lineterminator='\n')
        data.append(temp[['words', 'category_int']])
        del temp
        gc.collect()
    data = pd.concat(data)
    data = data.dropna()
    data['category_int'] -= 1
    data['category_int'] = data['category_int'].astype(int)
    return data


def sampling(data, sample_size):
    if sample_size > data.shape[0]:
        data_extra = data.sample(sample_size-data.shape[0])
        return pd.concat([data, data_extra])
    else:
        return data.sample(sample_size)


def sample_by_category(data, sample_size):
    if sample_size != -1:
        data= data.groupby('category_int').apply(sampling, sample_size).reset_index(drop=True)
    data = data.sample(frac=1).reset_index(drop=True)

    return data


def tokenize_data(data, tokenizer_path, maxlen):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(data['words'])
    data = data[['words_select', 'category_int']]
    data.columns = ['words', 'category_int']
    gc.collect()
    word_index = tokenizer.word_index
    with open(tokenizer_path, 'wb') as f:
        pickle.dump(tokenizer, f)
    sequences = tokenizer.texts_to_sequences(data['words'])
    labels = to_categorical(data['category_int'])
    data = pad_sequences(sequences, maxlen=maxlen, truncating='post', padding='post')
    logger.info('finished tokenizing data')
    return data, word_index, labels


def split_data(data, labels, validation_split, test_split):
    p1 = int(len(data)*(1-validation_split-test_split))
    p2 = int(len(data)*(1-test_split))
    x_train = data[:p1]
    y_train = labels[:p1]
    x_val = data[p1:p2]
    y_val = labels[p1:p2]
    x_test = data[p2:]
    y_test = labels[p2:]

    return x_train, y_train, x_val, y_val, x_test, y_test


def preprocess_for_training(data_paths, sample_size, tokenizer_path,maxlen, validation_split, test_split):
    data = import_data(data_paths, sample_size)
    data = sample_by_category(data, sample_size)
    data['words']=data['words'].astype(str)
    data['words_select'] = data['words'].apply(lambda x: ' '.join(x.split(' ')[0:maxlen]))
    data, word_index, labels = tokenize_data(data, tokenizer_path, maxlen)
    x_train, y_train, x_val, y_val, x_test, y_test = split_data(data, labels, validation_split, test_split)

    return x_train, y_train, x_val, y_val, x_test, y_test, word_index

