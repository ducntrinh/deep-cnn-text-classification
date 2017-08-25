#!/usr/bin/env python3

from argparse import ArgumentParser
import numpy as np
from gensim.models import KeyedVectors
from keras.preprocessing import sequence
from keras.callbacks import CSVLogger
from pymongo import MongoClient

from deep_text_cnn import DeepTextCNN

ARGS_PARSER = ArgumentParser()
ARGS_PARSER.add_argument(
    '-d', '--database', type=str, help='Name of database', required=True)
ARGS_PARSER.add_argument(
    '-e',
    '--embedding',
    type=str,
    help='Path to Google News word2vec pretrained file',
    required=True)
ARGS_PARSER.add_argument(
    '-n', '--numclasses', type=int, help='Number of classes', required=True)
ARGS = ARGS_PARSER.parse_args()

DATABASE_NAME = ARGS.database
EMBEDDING_FILE = ARGS.embedding
NUM_CLASSES = ARGS.numclasses

SEQUENCE_LENGTH = 200
VOCAB_SIZE = 30000
EMBEDDING_SIZE = 300
FILTER_SIZES = [3, 4, 5]
NUM_FILTERS = 150
DROPOUT_KEEP_PROB = float(0.5)

# Load database
MONGO_CLIENT = MongoClient()
DATABASE = MONGO_CLIENT.get_database(DATABASE_NAME)
TRAIN_COLLECTION = DATABASE.get_collection('train')
TEST_COLLECTION = DATABASE.get_collection('test')
VOCAB_COLLECTION = DATABASE.get_collection('vocab')

TRAIN_SIZE = TRAIN_COLLECTION.count()
TEST_SIZE = TEST_COLLECTION.count()


def rating_to_one_hot(rating):
    one_hot = np.zeros(NUM_CLASSES)
    one_hot[rating - 1] = 1
    return one_hot


def data_generator(collection, batch_size):
    holder_x = np.zeros((batch_size, SEQUENCE_LENGTH))
    holder_y = np.zeros((batch_size, NUM_CLASSES))
    k = 0

    while True:
        for document in collection.find({}, no_cursor_timeout=True):
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


def prepare_embedding():
    word2vec = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary=True)

    embedding_weights = np.zeros((VOCAB_SIZE, EMBEDDING_SIZE))
    for document in VOCAB_COLLECTION.find({}):
        word = document['word']
        index = document['index']
        if index >= VOCAB_SIZE:
            continue
        if word in word2vec.vocab:
            embedding_weights[index] = word2vec.word_vec(word)

    return embedding_weights


EMBEDDING_WEIGHTS = prepare_embedding()
MODEL = DeepTextCNN(SEQUENCE_LENGTH, NUM_CLASSES, VOCAB_SIZE, EMBEDDING_SIZE,
                    EMBEDDING_WEIGHTS, FILTER_SIZES, NUM_FILTERS,
                    DROPOUT_KEEP_PROB)

# Train
BATCH_SIZE = 128
EPOCHS = 20
STEPS_PER_EPOCH = int(TRAIN_SIZE / BATCH_SIZE)
VALIDATION_STEPS = int(TEST_SIZE / BATCH_SIZE)
print(TEST_SIZE)
print(VALIDATION_STEPS)


CSV_LOGGER = CSVLogger(DATABASE_NAME + '_log.csv', append=True, separator=',')
MODEL.summary()
MODEL.fit_generator(
    data_generator(TRAIN_COLLECTION, BATCH_SIZE),
    steps_per_epoch=STEPS_PER_EPOCH,
    epochs=EPOCHS,
    validation_data=data_generator(TEST_COLLECTION, BATCH_SIZE),
    validation_steps=VALIDATION_STEPS,
    callbacks=[CSV_LOGGER])
