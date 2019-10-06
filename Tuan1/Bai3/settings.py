import os

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_TRAIN_PATH = os.path.join(DIR_PATH, 'classify_data/train/')
DATA_TEST_PATH = os.path.join(DIR_PATH, 'classify_data/test/')
STOP_WORDS = os.path.join(DIR_PATH, 'data/vietnamese-stopwords.txt')
SPECIAL_CHARACTER = '0123456789%@$.,=+-!;/()*"&^:#|\n\t\''
