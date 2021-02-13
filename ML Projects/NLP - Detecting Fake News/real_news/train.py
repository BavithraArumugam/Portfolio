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

from nltk.tag import pos_tag
from nltk.stem.snowball import SnowballStemmer
import nltk
nltk.download('punkt')
from nltk.tag.stanford import StanfordNERTagger
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec
import numpy as np

fs = sys.argv[1]
model_name = sys.argv[2]

stemmer = SnowballStemmer("english")
##### FUNCTIONS #####
def tokenize_and_stem(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
            if re.search('[a-zA-Z]', token):
                    filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems

def tokenize_only(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
        tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
        filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
        for token in tokens:
            if re.search('[a-zA-Z]', token):
                    filtered_tokens.append(token)
        return filtered_tokens

#use regular expression to split by space
REGEX = re.compile(r"\s+")
def tokenize(text):
    return [tok.strip().lower() for tok in REGEX.split(text)]

#open file and read data
f = open(fs, newline='', encoding='utf-8')
reader = csv.reader(f, delimiter=',')

data = []
labels=[]
i = 0
toremove = {}
for row in reader:
        #remove non-ascii from the line
        if(i == 0):
            i = 1
            continue
        line = row[2]
        line = re.sub(r"[^\x00-\x7F]+", "", line)
        # line = re.sub(r"\n", "", line)
        # line = re.sub(r"\.", "", line)
        re.sub(r'[^\w]', ' ', line)
        line = tokenize_only(line)
        if(row[3] == 'FAKE'):
                labels.append(1)
        else:
                labels.append(0)
        data.append(line)
 
#size of data and labels
print(len(data))
print(len(labels))

#sample the data with labels 0 and 1
def subset(label_val):
  return [data[i] for i in np.where(np.asarray(labels)==label_val)[0]]
real_news = subset(0)
fake_news = subset(1)

#save model
model_real = Word2Vec(real_news, size=100, window=7, min_count=2, workers=4)
model_real.save(model_name)

print("model save in the name:", model_name)

