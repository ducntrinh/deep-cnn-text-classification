#!/usr/bin/env python3

# Import
from keras.preprocessing import sequence
from keras.callbacks import CSVLogger
from pymongo import MongoClient
import numpy as np

from deep_text_cnn import DeepTextCNN

# Load database
mongo_client = MongoClient()
database = mongo_client.get_database('yelpf')
train_collection = database.get_collection('train')
test_collection = database.get_collection('test')

SEQUENCE_LENGTH = 200
NUM_CLASSES = 2

train_size = train_collection.count()
test_size = test_collection.count() 

def rating_to_one_hot(rating):
    if rating > 3:
        return [1, 0]
    else:
        return [0, 1]


def data_generator(start, end, batch_size):
    holder_x = np.zeros((batch_size, SEQUENCE_LENGTH))
    holder_y = np.zeros((batch_size, NUM_CLASSES))
    k = 0

    while True:
        for i in range(start, end):
            document = collection.find_one({'_id': i})
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
FILTER_SIZE = 3
NUM_FILTERS = 150
NUM_UNITS = 150
DROPOUT_KEEP_PROB = float(0.5)

model = DeepTextCNN(SEQUENCE_LENGTH, NUM_CLASSES, VOCAB_SIZE, EMBEDDING_SIZE,
                  FILTER_SIZE, NUM_FILTERS, NUM_UNITS, DROPOUT_KEEP_PROB)

# Train
BATCH_SIZE = 128
EPOCHS = 10
steps_per_epoch = train_size / BATCH_SIZE
validation_steps = test_size / BATCH_SIZE

csv_logger = CSVLogger('log.csv', append=True, separator=';')
model.summary()
model.fit_generator(
    data_generator(train_start, train_end, BATCH_SIZE),
    steps_per_epoch=steps_per_epoch,
    epochs=EPOCHS,
    validation_data=data_generator(test_start, test_end, BATCH_SIZE),
    validation_steps=validation_steps,
    callbacks=[csv_logger])
