from textcls.preprocess.preprocess_clean_seg import segmentation_chunk
from argparse import ArgumentParser
import pandas as pd

parser = ArgumentParser()
parser.add_argument("-f", "--filename", help="optional argument", dest="filename")
args = parser.parse_args()
filename = args.filename
data = pd.read_csv(f'data/{filename}.csv')
data = segmentation_chunk(data)
data.to_csv(f'data/preprocessed/{filename}_preprocessed.csv', index = False)

