# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import gc 
import os
import re
import jieba
import time
from bs4 import BeautifulSoup
import sys
from textcls.utils.parallel import parallelize

def cleanMe(html):
    soup = BeautifulSoup(html) 
    for script in soup(["script", "style"]):
        script.extract()
    text = soup.get_text()
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = '\n'.join(chunk for chunk in chunks if chunk)
    return text

def segmentation(data):
    data = data.apply(lambda x: cleanMe(x))
    data = data.apply(lambda x: ' '.join(jieba.cut(x, HMM = True)))
    return data

def segmentation_chunk(data):
    data.columns = map(str.lower, data.columns)
    data['title'].replace(np.nan, '', inplace = True)
    data['content'].replace(np.nan, '', inplace = True)
    data['text'] = data[['title', 'content']].apply(lambda x: ''.join(x), axis=1)
    data['text'] = data['text'].str.replace('\n', '')
    data['text'] = data['text'].str.replace('\r', '')
    data['title_len'] = data['title'].str.len()
    data['content_len'] = data['content'].str.len()
    data['text_len'] = data['text'].str.len()
    jieba.set_dictionary('data/dict.txt.big')
    data['words'] = parallelize(data['text'], segmentation)
    return data

