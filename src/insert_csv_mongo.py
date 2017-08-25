#!/usr/bin/env python3

import pandas as pd
import re
from argparse import ArgumentParser
from pymongo import MongoClient
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from string import punctuation
from tqdm import tqdm

mongo_client = MongoClient(maxPoolSize=None)

pbar = None

tokenizer = None



def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        '-D',
        '--directory',
        type=str,
        help='Path to directory contain CSV files',
        required=True)
    parser.add_argument(
        '-d',
        '--database',
        type=str,
        help='Name of database to be inserted',
        required=True)
    parser.add_argument(
        '-n',
        '--numwords',
        type=int,
        help='Number of words in vocabulary',
        required=True)

    args = parser.parse_args()

    return args.directory, args.database, args.numwords


def get_collection(database_name, collection_name):
    return (mongo_client.get_database(database_name).get_collection(
        collection_name))


def clean_text(text, remove_stopwords=True, stem_words=True):
    # Convert words to lower case and split them
    text = text.lower().split()

    # Optionally, remove stop words
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]

    text = " ".join(text)

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r";", " ", text)
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
    return (text)


def clean_and_insert(directory, database_name):
    clean_and_insert_with_collection(directory, database_name, 'train')
    clean_and_insert_with_collection(directory, database_name, 'test')


def clean_and_insert_with_collection(directory, database_name, collection_name):
    file_path = directory + '/' + collection_name + '.csv'
    collection = get_collection(database_name, collection_name)

    reader = pd.read_csv(file_path, header=None, chunksize=32, iterator=True)
    progress_bar = tqdm(
        desc='Clean and insert in collection {}'.format(collection_name),
        unit=' items')
    document_id = 0
    document_holder = []
    for chunk in reader:
        data_frame = pd.DataFrame(chunk)
        for _, row in data_frame.iterrows():
            raw_text = str(row[1])
            cleaned_text = clean_text(raw_text)

            # Insert to database
            mongo_document = {
                '_id': document_id,
                'raw': raw_text,
                'cleaned': cleaned_text,
                'rating': int(row[0])
            }
            document_holder.append(mongo_document)
            # collection.insert_one(mongo_document)

            if len(document_holder) % 10000 == 0:
                collection.insert_many(document_holder)
                document_holder = []

            # Update progress bar
            progress_bar.update()

            # Increase document
            document_id += 1

    if len(document_holder) != 0:
        collection.insert_many(document_holder)

    progress_bar.close()

def cleaned_text_generator(database_name, collection_name):
    global pbar
    collection = get_collection(database_name, collection_name)
    for document in collection.find({}):
        pbar.update()
        yield document['cleaned']

def fit_texts_to_tokenizer(database_name):
    global pbar
    pbar = tqdm(
        desc='Fit cleaned text on collection train',
        unit=' items')
    tokenizer.fit_on_texts(cleaned_text_generator(database_name, 'train'))
    pbar.close()

    pbar = tqdm(
        desc='Fit cleaned text on collection test',
        unit=' items')
    tokenizer.fit_on_texts(cleaned_text_generator(database_name, 'test'))
    pbar.close()


def insert_vocabulary(database_name):
    collection = get_collection(database_name, 'vocab')
    for word, index in tokenizer.word_index.items():
        collection.insert_one({'word': word, 'index': index})


def turn_sequence_and_insert(database_name):
    turn_sequence_and_insert_with_collection(database_name, 'train')
    turn_sequence_and_insert_with_collection(database_name, 'test')


def turn_sequence_and_insert_with_collection(database_name, collection_name):
    collection = get_collection(database_name, collection_name)

    progress_bar = tqdm(
        desc='Turn text to sequence in collection {}'.format(collection_name),
        unit=' items')
    for document in collection.find({}, snapshot=True):
        sequence = tokenizer.texts_to_sequences([document['cleaned']])
        collection.update_one({
            '_id': document['_id']
        }, {'$set': {
            'sequence': sequence[0]
        }})
        progress_bar.update()

    progress_bar.close()


def main():
    directory, database_name, num_words = get_args()

    from keras.preprocessing.text import Tokenizer
    global tokenizer
    tokenizer = Tokenizer(num_words=num_words)

    clean_and_insert(directory, database_name)
    fit_texts_to_tokenizer(database_name)
    insert_vocabulary(database_name)
    turn_sequence_and_insert(database_name)


if __name__ == '__main__':
    main()
