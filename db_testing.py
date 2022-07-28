import datetime

import pandas as pd
from pymongo import MongoClient


mongo_client = MongoClient('localhost', 27017)

db = mongo_client.projeto_wise

df = pd.read_csv('dataset_customer_churn.csv', sep='^').to_dict('records')

clientes_collection = db.TESTE_INSERT

# def insert_dataset():

if __name__ == '__main__':
    clientes_collection.insert_many(df)