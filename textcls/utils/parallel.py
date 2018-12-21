import pandas as pd
from multiprocessing import cpu_count, Pool
import gc
 
cores = cpu_count()
partitions = cores

def split_index(index, num):
	size = index.shape[0]//num
	[(i, size+ i*size) for i in num]

def parallelize(data, func, **kwargs):
    size = data.shape[0]//partitions
    data_split = []
    for i in range(partitions):
    	if( i == partitions -1):
    		data_split.append(data.loc[i*size:])
    	else:
    		data_split.append(data.loc[i*size:i*size+size-1])
    del data
    gc.collect()
    pool = Pool(cores)
    if kwargs:
        data_split = [(data, kwargs) for data in data_split]
        data = pool.starmap(func, data_split)
    else:
        data = pool.map(func, data_split)
    del data_split
    gc.collect()
    data = pd.concat(data)
    pool.close()
    pool.join()
    return data
