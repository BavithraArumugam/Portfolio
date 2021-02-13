from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import re
import sys
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn import metrics
from sklearn.pipeline import make_pipeline
import operator
from sklearn import svm
from sklearn.model_selection import cross_val_score

import csv
from keras.models import load_model

from nltk.tag import pos_tag
from nltk.stem.snowball import SnowballStemmer
import nltk
from nltk.tag.stanford import StanfordNERTagger
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec
import numpy as np

test_file = open(sys.argv[1], newline='', encoding='utf-8')
model_name = sys.argv[2]

words=test_file.read()[:-1].split(',')
print(words)
model_fake= Word2Vec.load(model_name)

print("###### Top 5 similar words in FAKE-NEWS of ####")
for word in words:
	print(word,"is:")
	print(*model_fake.wv.similar_by_word(word.lower(),topn=5, restrict_vocab=None), sep="\n")
	print("\n")

