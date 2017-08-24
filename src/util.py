#!/usr/bin/env python3
import re
import numpy as np

from pymongo import MongoClient

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from string import punctuation

from gensim.models import KeyedVectors
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# Constant
DATABASE_NAME = 'yelpf'
EMBEDDING_FILE = '/data/personal/deep-cnn-text-classification/dataset/GoogleNews-vectors-negative300.bin'
MAX_NUMBER_WORDS = 30000

# Load database
mongo_client = MongoClient()
database = mongo_client.get_database(DATABASE_NAME)
train_collection = database.get_collection('train')
test_collection = database.get_collection('test')

# Load Google News word2vec
# word2vec = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary=True)
# print('Loaded {} word vectors of Google News word2vec'.format(len(word2vec.vocab)))

def text_to_wordlist(text, remove_stopwords=True, stem_words=True):
    # Clean the text, with the option to remove stopwords and to stem words.

    # Convert words to lower case and split them
    text = text.lower().split()

    # Optionally, remove stop words
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]

    text = " ".join(text)

    # Clean the text text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)

    # Optionally, shorten words to their stems
    if stem_words:
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)

    # Return a list of words
    return(text)


def wordlist_generator(collection):
    for document in collection.find():
        wordlist = text_to_wordlist(document['review'])
        yield wordlist

tokenizer = Tokenizer(num_words=MAX_NUMBER_WORDS)
tokenizer.fit_on_texts(wordlist_generator(train_collection))
tokenizer.fit_on_texts(wordlist_generator(test_collection))
word_index = tokenizer.word_index
print('Found {} unique tokens'.format(len(word_index)))

def update_database(collection):
    for document in collection.find(snapshot=True):
        document['sequence'] = text_to_wordlist(document['review'])
        collection.replace_one({'_id': document['_id']}, document)

update_database(train_collection)
update_database(test_collection)
