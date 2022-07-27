import datetime

import pandas as pd
from pymongo import MongoClient

file = open('dataset_customer_churn.csv', 'r')
file.write(file_text.replace('^', ';'))
file_text = file.read()

mongo_client = MongoClient('localhost', 27017)




db = mongo_client.projeto_wise

clientes_collection = db.teste

if __name__ == '__main__':
    print(file_text)