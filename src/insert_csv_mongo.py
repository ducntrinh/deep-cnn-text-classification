#!/usr/bin/env python3

import progressbar
import pandas as pd
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from pymongo import MongoClient

mongo_client = MongoClient(maxPoolSize=None)

widgets = [
    progressbar.RotatingMarker(), progressbar.Counter('Inserted: %(value)05d'),
    ' reviews (', progressbar.Timer(), ')'
]
bar = progressbar.ProgressBar(
    max_value=progressbar.UnknownLength, widgets=widgets)

executor = ThreadPoolExecutor(max_workers=8)


def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        '-f', '--file', type=str, help='Path to CSV file', required=True)
    parser.add_argument(
        '-d',
        '--database',
        type=str,
        help='Name of database to be inserted',
        required=True)
    parser.add_argument(
        '-c',
        '--collection',
        type=str,
        help='Name of collection to be inserted',
        required=True)

    args = parser.parse_args()

    return args.file, args.database, args.collection


def insert_document(document, database_name, collection_name):
    mongo_client.get_database(database_name).get_collection(
        collection_name).insert_one(document)
    bar.update(document['_id'] + 1)


def main():
    file_path, database_name, collection_name = get_args()

    reader = pd.read_csv(file_path, header=None, chunksize=32, iterator=True)
    document_id = 0
    for chunk in reader:
        df = pd.DataFrame(chunk)
        for index, row in df.iterrows():
            document = {
                '_id': document_id,
                'review': row[1],
                'rating': int(row[0])
            }
            executor.submit(insert_document, document, database_name,
                            collection_name)
            # insert_document(document, database_name, collection_name)
            document_id += 1
    executor.shutdown()
    bar.finish()


if __name__ == '__main__':
    main()
