#!/usr/bin/env python3

# Import
import numpy as np
from gensim.models import KeyedVectors
from keras.preprocessing import sequence
from keras.callbacks import CSVLogger
from pymongo import MongoClient

from deep_text_cnn import DeepTextCNN

# Load database
mongo_client = MongoClient()
database = mongo_client.get_database('test')
train_collection = database.get_collection('train')
test_collection = database.get_collection('test')
vocab_collection = database.get_collection('vocab')

EMBEDDING_FILE = '/home/ductn/Downloads/GoogleNews-vectors-negative300.bin'
SEQUENCE_LENGTH = 200
NUM_CLASSES = 5

TRAIN_SIZE = train_collection.count()
TEST_SIZE = test_collection.count()


def rating_to_one_hot(rating):
    one_hot = np.zeros(NUM_CLASSES)
    one_hot[rating - 1] = 1
    return one_hot


def data_generator(collection, batch_size):
    holder_x = np.zeros((batch_size, SEQUENCE_LENGTH))
    holder_y = np.zeros((batch_size, NUM_CLASSES))
    k = 0

    while True:
        for document in collection.find({}):
            x = sequence.pad_sequences([document['sequence']], SEQUENCE_LENGTH)
            y = np.array([rating_to_one_hot(document['rating'])])

            if k != 0 and k % batch_size == 0:
                k = 0
                yield holder_x, holder_y
                holder_x = np.zeros((batch_size, SEQUENCE_LENGTH))
                holder_y = np.zeros((batch_size, NUM_CLASSES))

            holder_x[k] = x
            holder_y[k] = y
            k += 1


# Create model
VOCAB_SIZE = 30000
EMBEDDING_SIZE = 300
FILTER_SIZES = [3, 4, 5]
NUM_FILTERS = 150
DROPOUT_KEEP_PROB = float(0.5)


def prepare_embedding():
    word2vec = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary=True)

    embedding_weights = np.zeros((VOCAB_SIZE, EMBEDDING_SIZE))
    for document in vocab_collection.find({}):
        word = document['word']
        index = document['index']
        if index > VOCAB_SIZE:
            continue
        if word in word2vec.vocab:
            embedding_weights[index] = word2vec.word_vec(word)

    return embedding_weights


EMBEDDING_WEIGHTS = prepare_embedding()
model = DeepTextCNN(SEQUENCE_LENGTH, NUM_CLASSES, VOCAB_SIZE, EMBEDDING_SIZE,
                    EMBEDDING_WEIGHTS, FILTER_SIZES, NUM_FILTERS,
                    DROPOUT_KEEP_PROB)

# Train
BATCH_SIZE = 128
EPOCHS = 10
STEPS_PER_EPOCH = TRAIN_SIZE / BATCH_SIZE
VALIDATION_STEPS = TEST_SIZE / BATCH_SIZE

CSV_LOGGER = CSVLogger('log.csv', append=True, separator=';')
model.summary()
model.fit_generator(
    data_generator(train_collection, BATCH_SIZE),
    steps_per_epoch=STEPS_PER_EPOCH,
    epochs=EPOCHS,
    validation_data=data_generator(test_collection, BATCH_SIZE),
    validation_steps=VALIDATION_STEPS,
    callbacks=[CSV_LOGGER])
