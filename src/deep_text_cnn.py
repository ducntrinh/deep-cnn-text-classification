from keras import initializers
from keras.layers import Input, Dense, Embedding, Conv1D, MaxPooling1D, Dropout
from keras.layers.core import Reshape, Flatten
from keras.layers.merge import Concatenate
from keras.optimizers import Adam
from keras import regularizers
from keras.models import Model
import keras.backend as K
from sklearn import metrics


def DeepTextCNN(sequence_length, num_classes, vocab_size, embedding_size,
                embedding_weights, filter_sizes, num_filters,
                dropout_keep_prob):
    # Input layer
    inputs = Input(shape=(sequence_length, ), dtype='int32')

    # Embedding layer
    if embedding_weights is not None:
        embedding = Embedding(
            output_dim=embedding_size,
            input_dim=vocab_size,
            input_length=sequence_length,
            weights=[embedding_weights])(inputs)
    else:
        embedding = Embedding(
            output_dim=embedding_size,
            input_dim=vocab_size,
            input_length=sequence_length,
            embeddings_initializer='uniform')(inputs)
    # reshape = Reshape((sequence_length, embedding_size, 1))(embedding)

    # Convolutional & max pooling layers
    pooled_outputs = []
    for i, filter_size in enumerate(filter_sizes):
        conv = Conv1D(
            filters=100,
            kernel_size=filter_size,
            activation='relu',
            padding='same',
            kernel_regularizer=regularizers.l2(0.0001))(embedding)
        pooled = MaxPooling1D(pool_size=2)(conv)
        dropout_pooled = Dropout(0.1)(pooled)
        pooled_outputs.append(dropout_pooled)

    # Merge pooled output
    concat = Concatenate(axis=2)(pooled_outputs)

    # Convolutional & max pool layers
    conv1 = Conv1D(
        filters=100, kernel_size=5, activation='relu', kernel_regularizer=regularizers.l2(0.0001))(concat)
    pool1 = MaxPooling1D(pool_size=5)(conv1)
    dropout_pool1 = Dropout(0.2)(pool1)
    conv2 = Conv1D(
        filters=100, kernel_size=5, activation='relu', kernel_regularizer=regularizers.l2(0.0001))(dropout_pool1)
    pool2 = MaxPooling1D(pool_size=5)(conv2)

    flatten = Flatten()(pool2)

    # Add dropout
    dropout = Dropout(dropout_keep_prob)(flatten)

    # Output layer
    outputs = Dense(activation='softmax', units=num_classes, kernel_regularizer=regularizers.l2(0.0001))(dropout)

    # Create model from input and output
    model = Model(inputs=inputs, outputs=outputs)

    # Optimizer
    adam = Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    # Compile model
    model.compile(
        optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

    return model
