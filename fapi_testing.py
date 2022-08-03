from cmath import inf
from urllib import request
from fastapi import FastAPI, Request
from pymongo import MongoClient
import pandas as pd

mongo_client = MongoClient('localhost', 27017)

clientes_collection = mongo_client.projeto_trainee.clientes

data = pd.DataFrame(list(clientes_collection.find({})))
data['_id'] = data['_id'].map(str)
data.drop(data[data.isna().any(axis=1)].index, axis=0, inplace=True)

fast_api = FastAPI()

@fast_api.get("/db_returning")
def db_returning():
    return data.sample(50).to_dict('records')

@fast_api.get("/")
def home():
    welcome_message = "Welcome to iml api!\nHere, you can a whole lot of stuff, that i will sometime later add"
    return {"response":welcome_message}

@fast_api.post('/teste_post')
async def teste_post(info:Request):
    request_json = await info.json()
    return {'status':"deu boa haha", "data":request_json}