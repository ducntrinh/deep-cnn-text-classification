#!/usr/bin/env python3

from pymongo import MongoClient
import csv
from concurrent.futures import ThreadPoolExecutor

def insert_row(id, review, rating, collection):
   collection.insert_one({'_id': id, 'review': review, 'rating': rating}) 

def main():
    file_path = '/data/personal/deep-cnn-text-classification/dataset/yelp_review_full_csv/train.csv'
    database_name = 'yelpf'
    collection_name = 'train'

    mongo_client = MongoClient()
    database = mongo_client.get_database(database_name)
    collection = database.get_collection(collection_name)
    executor = ThreadPoolExecutor()
    with open(file_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        id = 0
        for row in csv_reader:
            executor.submit(insert_row, id, row[1], int(row[0]), collection)
            id += 1

if __name__ == '__main__':
    main()
