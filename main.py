import time
import pandas as pd
import numpy as np
from fastapi import FastAPI, Request
from scipy.stats import randint
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score, accuracy_score, precision_score
from sklearn.model_selection import cross_validate, KFold, RandomizedSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from pymongo import MongoClient
import datetime

def train_test_spliting(x_param, y_param):
    train_x_, test_x_, train_y_, test_y_ = train_test_split(x_param, y_param, test_size=0.2, stratify=y_param)
    return train_x_, test_x_, train_y_, test_y_

def standardize_data_and_split(x_param, y_param):
    stdscaler = StandardScaler()
    std_x = stdscaler.fit_transform(x_param)
    std_train_x, std_test_x, train_y_, test_y_ = train_test_spliting(std_x, y_param)
    return std_train_x, std_test_x, train_y_, test_y_


def fast_api_print(text):
    return f"\033[92mINFO\033[0m:     {text}"


def remove_NANs_and_outliers(data_param):
    data_param.drop(data_param[data_param.isna().any(axis=1)].index, axis=0, inplace=True)
    data_param.drop(data_param[data_param.QTDE_DIAS_ATIVO > 100000].index, axis=0, inplace=True)
    data_param.drop(data_param[data_param['QTDE_DIAS_ATIVO'] == 1790].index, inplace=True, axis=0)
    return data_param


def data_setting(data_param):
    data_param_sanitized = data_sanitization(data_param)
    data_dummies = pd.get_dummies(data_param_sanitized.drop(labels='CLIENTE', axis=1))
    x_ = data_dummies.drop(labels=['SITUACAO'], axis=1)
    y_ = data_dummies['SITUACAO']
    return data_dummies, x_, y_, data_param_sanitized


def data_sanitization(data_param):
    new_data_sanitized = data_param.copy()
    new_data_sanitized.drop(labels='_id', axis=1, inplace=True)
    dict_replace = {
        "SIM": 1,
        "NAO": 0,
        'F': 0,
        'M': 1,
        'DESATIVADO': 1,
        'ATIVO': 0,

    }
    new_data_sanitized.replace(dict_replace, inplace=True)
    QTDE_DIAS_ATIVO_MENOR_QUE_365 = np.array(new_data_sanitized['QTDE_DIAS_ATIVO'] < 365)
    new_data_sanitized['QTDE_DIAS_ATIVO_MENOR_QUE_365'] = 0
    new_data_sanitized.loc[QTDE_DIAS_ATIVO_MENOR_QUE_365, 'QTDE_DIAS_ATIVO_MENOR_QUE_365'] = 1
    QTDE_DIAS_ATIVO_MENOR_QUE_1000 = np.array(
        (new_data_sanitized['QTDE_DIAS_ATIVO'] >= 365) & (data_param['QTDE_DIAS_ATIVO'] < 1000))
    new_data_sanitized['QTDE_DIAS_ATIVO_MENOR_QUE_1000'] = 0
    new_data_sanitized.loc[QTDE_DIAS_ATIVO_MENOR_QUE_1000, 'QTDE_DIAS_ATIVO_MENOR_QUE_1000'] = 1
    QTDE_DIAS_ATIVO_MAIOR_QUE_1000 = np.array(data_param['QTDE_DIAS_ATIVO'] >= 1000)
    new_data_sanitized['QTDE_DIAS_ATIVO_MAIOR_QUE_1000'] = 0
    new_data_sanitized.loc[QTDE_DIAS_ATIVO_MAIOR_QUE_1000, 'QTDE_DIAS_ATIVO_MAIOR_QUE_1000'] = 1
    new_data_sanitized.drop(labels='QTDE_DIAS_ATIVO', inplace=True, axis=1)
    new_data_sanitized.drop(
        labels=['A006_REGISTRO_ANS', 'CODIGO_BENEFICIARIO', 'CD_USUARIO', 'CODIGO_FORMA_PGTO_MENSALIDADE',
                'A006_NM_PLANO', 'CD_ASSOCIADO', 'ESTADO_CIVIL'], axis=1,
        inplace=True)
    return new_data_sanitized


np.random.seed(1010)

MODELS = []
MODELS_INFO = {}
PREDICTIONS = {}

app = FastAPI()

print(fast_api_print("Connecting to MongoDB."))
mongo_client = MongoClient('localhost', 27017)
db = mongo_client.projeto_trainee
clientes_collection = db.clientes

print(fast_api_print("Getting Dataframe."))
data = pd.DataFrame(list(clientes_collection.find({})))
data = remove_NANs_and_outliers(data)

print(fast_api_print("Setting X,y and data's."))
data_dummified, x, y, data_sanitized = data_setting(data)
train_x, test_x, train_y, test_y = standardize_data_and_split(x, y)



def scores(model_param):
    model_predicts = model_param.predict(test_x)
    model_scores_dict = {
        'Accuracy score': accuracy_score(test_y, model_predicts),
        'Recall score': recall_score(test_y, model_predicts),
        'Precision score': precision_score(test_y, model_predicts),
        'F1 score': f1_score(test_y, model_predicts),
        'Overall score': ((accuracy_score(test_y, model_predicts) + recall_score(test_y, model_predicts) +
                           precision_score(test_y, model_predicts) + f1_score(test_y, model_predicts)) / 4)
    }
    return model_scores_dict



def randomForest_model():
    time_on_creation = time.time()
    print(fast_api_print('Creating Random Forest Model.'))
    random_forest_classifier = RandomForestClassifier(n_estimators=10)
    print(fast_api_print('Fitting data into model for training.'))
    random_forest_classifier.fit(train_x, train_y)
    time_to_finish = time.time() - time_on_creation
    print(fast_api_print('Done!'))

    model_info = {
        'model_index': (len(MODELS)),
        'model_type': random_forest_classifier.__class__.__name__,
        'model_parameters': random_forest_classifier.get_params(),
        'seconds_to_create_model': round(time_to_finish, 2),
        'model_scores': scores(random_forest_classifier),
        'model_creation_data': datetime.datetime.now()
    }

    return {'model_info': model_info, 'model': random_forest_classifier}


def logisticRegression_model():
    time_on_creation = time.time()
    print(fast_api_print('Creating Logistic Regression Model.'))
    log_regression = LogisticRegression(max_iter=20000)
    print(fast_api_print('Fitting data into model for training.'))
    log_regression.fit(train_x, train_y)
    time_to_finish = time.time() - time_on_creation
    print(fast_api_print('Done!'))
    model_info = {
        'model_index': (len(MODELS)),
        'model_type': log_regression.__class__.__name__,
        'model_parameters': log_regression.get_params(),
        'seconds_to_create_model': round(time_to_finish, 2),
        'model_scores': scores(log_regression),
        'model_creation_data': datetime.datetime.now()
    }

    return {'model_info': model_info, 'model': log_regression}


def multilayerPerceptron_model():
    time_on_creation = time.time()
    print(fast_api_print('Creating MultiLayer Perceptron Model.'))
    mlp_classifier = MLPClassifier()
    print(fast_api_print('Fitting data into model for training.'))
    mlp_classifier.fit(train_x, train_y)
    time_to_finish = time.time() - time_on_creation
    print(fast_api_print('Done!'))
    model_info = {
        'model_index': (len(MODELS)),
        'model_type': mlp_classifier.__class__.__name__,
        'model_parameters': mlp_classifier.get_params(),
        'seconds_to_create_model': round(time_to_finish, 2),
        'model_scores': scores(mlp_classifier),
        'model_creation_data': datetime.datetime.now()
    }

    return {'model_info': model_info, 'model': mlp_classifier}


def model_scores():
    model_scores_dict = {}
    index_model = 0
    for model_ in MODELS:
        model_scores_dict[str(index_model) + " - " + model_.__class__.__name__] = scores(model_)
        index_model += 1
    if not model_scores_dict:
        return 'No models created!'
    else:
        return model_scores_dict


def cross_validation(cv_options):
    try:
        model_index = cv_options['model_choice_index']
    except KeyError:
        return 'Please, send a JSON with {model_choice_index:(index of model choice)} on POST requisition.'
    except TypeError:
        return 'Please, send a JSON with {model_choice_index:(index of model choice)} on POST requisition.'

    try:
        n_splits = cv_options['n_splits']
    except KeyError:
        n_splits = 5
    try:
        shuffle = cv_options['shuffle']
    except KeyError:
        shuffle = True
    try:
        print(fast_api_print(f"Doing Cross validation on model {MODELS[model_index].__class__.__name__}"))
        validation_results = cross_validate(MODELS[model_index], x, y,
                                            cv=KFold(n_splits=n_splits, shuffle=shuffle))
        return pd.DataFrame(validation_results).to_dict('records')
    except IndexError as e:
        return f'Index out of range.\n you have {len(MODELS)} models created.\n{e}'
    except ValueError as e:
        return f'Invalid Key on JSON file!\n{e}'


def clustering(n_clusters_choice):
    try:
        n_clusters = n_clusters_choice["n_clusters"]
    except TypeError:
        n_clusters = 5
    except KeyError:
        n_clusters = 5
    stdscaler = StandardScaler()
    data_scaled = stdscaler.fit_transform(data_dummified)
    try:
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, max_iter=300, random_state=1010)
        kmeans.fit(data_scaled)
        data_dummified['CLUSTER'] = kmeans.labels_
        description = data_dummified.groupby('CLUSTER')
        n_clients = description.size()
        description = description.mean()
        description['n_clients'] = n_clients
        description_json = description.to_dict('records')
        return description_json
    except TypeError as e:
        return f"n_clusters_choice must be a integer!\n{e}"


def hyper_parameters_multilayer_perceptron():
    RSCV_parameters = {
        'hidden_layer_sizes': [(50, 50, 50), (50, 100, 50), (100,)],
        'activation': ['tanh', 'relu'],
        'solver': ['sgd', 'adam'],
        'alpha': [0.0001, 0.05],
        'learning_rate': ['constant', 'adaptive']
    }
    print(fast_api_print(
        "Doing the Multilayer Perceptron Classifier Hyper Parameters through Randomized Search Cross Validation... (this might take a while)"))
    RSCross_validation = RandomizedSearchCV(MLPClassifier(),
                                            RSCV_parameters, n_iter=10, cv=KFold(n_splits=5, shuffle=True))
    RSCross_validation.fit(train_x, train_y)
    print(fast_api_print('Done!'))
    results = pd.DataFrame(RSCross_validation.cv_results_)
    results_html = results.sort_values(by='rank_test_score').to_html()
    results_html_file = open("hyper parameters results - MLPClassifier.html", "w")
    results_html_file.write(results_html)
    results_html_file.close()
    hp_best_estimator = RSCross_validation.best_estimator_
    MODELS.append(hp_best_estimator)
    model_info = {
        'model_type': hp_best_estimator.__class__.__name__,
        'model_parameters': hp_best_estimator.get_params(),
        'model_scores': scores(hp_best_estimator)
    }

    return {'model_info': model_info, 'model': hp_best_estimator}


def hyper_parameters_logistic_regression():
    RSCV_parameters = {
        'solver': ['newton-cg', 'lbfgs', 'liblinear'],
        'penalty': ['l2'],
        'C': [100, 10, 1.0, 0.1, 0.01]
    }
    print(fast_api_print(
        "Doing the Logistic Regression Hyper Parameters through Randomized Search Cross Validation... (this might take a while)"))
    RSCross_validation = RandomizedSearchCV(LogisticRegression(max_iter=20000),
                                            RSCV_parameters, n_iter=10, cv=KFold(n_splits=5, shuffle=True))
    RSCross_validation.fit(train_x, train_y)
    print(fast_api_print('Done!'))
    results = pd.DataFrame(RSCross_validation.cv_results_)
    results_html = results.sort_values(by='rank_test_score').to_html()
    results_html_file = open("hyper parameters results - LRC.html", "w")
    results_html_file.write(results_html)
    results_html_file.close()
    print(fast_api_print(
        'A HTML file was created with the results dataframe, ordered by their score! (incase you want to review all the models)'))
    hp_best_estimator = RSCross_validation.best_estimator_
    MODELS.append(hp_best_estimator)
    model_info = {
        'model_type': hp_best_estimator.__class__.__name__,
        'model_parameters': hp_best_estimator.get_params(),
        'model_scores': scores(hp_best_estimator)
    }

    return {'model_info': model_info, 'model': hp_best_estimator}


def hyper_parameters_random_forest():
    RSCV_parameters = {
        "max_depth": randint(10, 250),
        "min_samples_split": randint(2, 16),
        "min_samples_leaf": randint(1, 16),
        "bootstrap": [True, False],
        "criterion": ["gini", "entropy"]
    }
    print(fast_api_print(
        "Doing the Random Forest Classifier Hyper Parameters through Randomized Search Cross Validation... (this might take a while)"))
    RSCross_validation = RandomizedSearchCV(RandomForestClassifier(n_estimators=10), RSCV_parameters,
                                            n_iter=10, cv=KFold(n_splits=5, shuffle=True))
    RSCross_validation.fit(train_x, train_y)
    results = pd.DataFrame(RSCross_validation.cv_results_)
    results_html = results.sort_values(by='rank_test_score').to_html()
    results_html_file = open('hyper parameters results - RFC.html', 'w')
    results_html_file.write(results_html)
    hp_best_estimator = RSCross_validation.best_estimator_

    MODELS.append(hp_best_estimator)
    model_info = {
        'model_type': hp_best_estimator.__class__.__name__,
        'model_parameters': hp_best_estimator.get_params(),
        'model_scores': scores(hp_best_estimator)
    }

    return {'model_info': model_info, 'model': hp_best_estimator}


def classify_user(column, column_info, user_id_column):
    model_predictions = {}
    model_index = 0
    indexes = data[data[column] == column_info].index
    for model in MODELS:
        predictions = model.predict(x.loc[indexes].values)
        predictions_proba = model.predict_proba(x.loc[indexes].values)
        user_ids = data.loc[indexes, user_id_column].to_numpy()
        prediction_dataframe = pd.DataFrame()
        prediction_dataframe['ID'] = user_ids
        prediction_dataframe['PREDICTIONS'] = predictions
        get_proba = [round(proba[1], 3) for proba in predictions_proba]
        prediction_dataframe['CHURN_PROBABILITY'] = get_proba
        model_predictions[str(model_index) + " - " + model.__class__.__name__] = prediction_dataframe.to_dict('records')
        model_index += 1
    save_predictions({"modo":"column_equals", "valores":model_predictions})
    return model_predictions


def classify_all(user_id_column, model_choice_index):
    model_predictions = {}
    model = MODELS[model_choice_index]
    predictions = model.predict(x)
    predictions_proba = model.predict_proba(x)
    user_ids = data.loc[data.index, user_id_column]
    prediction_dataframe = pd.DataFrame()
    prediction_dataframe['ID'] = user_ids
    prediction_dataframe['PREDICTIONS'] = predictions
    get_proba = [round(proba[1], 3) for proba in predictions_proba]
    prediction_dataframe['CHURN_PROBABILITY'] = get_proba
    model_predictions[str(model_choice_index) + " - " + model.__class__.__name__] = prediction_dataframe.to_dict('records')
    save_predictions({"modo":"all", "valores":prediction_dataframe.to_dict('records')})
    return model_predictions


def save_predictions(predictions):
    PREDICTIONS[f'{len(PREDICTIONS)} - {datetime.datetime.now()} - {predictions["modo"]}'] = predictions['valores']
    return 'Prediction saved.'


@app.get('/')
async def home():
    welcome_message = "Welcome to iml api!\nHere, you can a whole lot of stuff, that i will sometime later add" \
                      "\n the api is focused on Machine Learning and model creation!"
    return welcome_message


@app.get('/create_model/RFC')
async def api_create_model_RFC(request:Request):
    print(fast_api_print(f'{request.client.host} requested a RFC creation.'))
    model = randomForest_model()
    MODELS_INFO[str(len(MODELS)) + " - " + model['model'].__class__.__name__] = model['model_info']
    MODELS.append(model['model'])
    return model['model_info']


@app.get('/create_model/LRC')
async def api_create_model_LRC(request:Request):
    print(fast_api_print(f'{request.client.host} requested a LRC creation.'))
    model = logisticRegression_model()
    MODELS.append(model['model'])
    MODELS_INFO[str(len(MODELS)) + " - " + model['model'].__class__.__name__] = model['model_info']
    return model['model_info']


@app.get('/create_model/MLP')
async def api_create_model_MLP(request:Request):
    print(fast_api_print(f'{request.client.host} requested a MLP creation.'))
    model = multilayerPerceptron_model()
    MODELS.append(model['model'])
    MODELS_INFO[str(len(MODELS)) + " - " + model['model'].__class__.__name__] = model['model_info']
    return model['model_info']


@app.get('/model_visualization/')
async def api_model_visualization(request:Request):
    print(fast_api_print(f'{request.client.host} requested to view all models.'))
    return MODELS_INFO


@app.post('/set_data/')
async def api_set_data(request:Request):
    print(fast_api_print(f'{request.client.host} requested to set a new Data and X,y.'))
    file_loc_json = await request.json()
    data_dummified = pd.read_csv(file_loc_json['file_loc'])
    x = data_dummified.drop(labels=[file_loc_json['y']], axis=1)
    y = data_dummified[file_loc_json['y']]
    train_x, test_x, train_y, test_y = standardize_data_and_split(x, y)
    return 'new data was set!'


@app.get('/model_score/')
async def api_model_scores(request:Request):
    print(fast_api_print(f'{request.client.host} requested to view all model scores.'))
    return model_scores()


@app.get('/create_model/hyper_parameters/RFC')
async def api_create_model_hp_RFC(request:Request):
    print(fast_api_print(f'{request.client.host} requested a RFC model creation through HP.'))
    return hyper_parameters_random_forest()['model_info']


@app.get('/create_model/hyper_parameters/LRC')
async def api_create_model_hp_LRC(request:Request):
    print(fast_api_print(f'{request.client.host} requested a LRC model creation through HP.'))
    return hyper_parameters_logistic_regression()['model_info']


@app.get('/create_model/hyper_parameters/MLP')
async def api_create_model_hp_MLP(request:Request):
    print(fast_api_print(f'{request.client.host} requested a MLP model creation through HP.'))
    return hyper_parameters_multilayer_perceptron()['model_info']


@app.post('/cross_validation/')
async def api_cross_validation(request:Request):
    print(fast_api_print(f'{request.client.host} requested to cross-validate a model.'))
    cv_options = await request.json()
    return cross_validation(cv_options)


@app.post('/clustering/')
async def api_clustering(request:Request):
    print(fast_api_print(f'{request.client.host} requested to clusterize the onehot-encoded data.'))
    n_clusters = await request.json()
    return clustering(n_clusters)


@app.post('/save_sanitized_dataset_mongo/')
async def api_sanitized_dataset_saving(request:Request):
    print(fast_api_print(f'{request.client.host} requested to save the current sanitized dataset on MongoDB'))
    request_json = await request.json()
    data_to_dict = data_sanitized.to_dict('records')
    if list(db[request_json['collection_name']].find({})):
        return 'A collection with this name already exists, please, choose another. (or use ' \
               '"/save_dataset_mongo/force" to replace.) '
    db[request_json['collection_name']].insert_many(data_to_dict)
    return f'Data was saved into Mongodb with the name {request_json["collection_name"]}.'


@app.post('/save_sanitized_dataset_mongo/force')
async def api_sanitized_dataset_saving_force(request:Request):
    print(fast_api_print(f'{request.client.host} requested to save the current sanitized dataset on MongoDB (Forced)'))
    request_json = await request.json()
    data_to_dict = data_sanitized.to_dict('records')
    if list(db[request_json['collection_name']].find({})):
        data_set_dropped = True
        db[request_json['collection_name']].drop()
    else:
        data_set_dropped = False

    db[request_json['collection_name']].insert_many(data_to_dict)
    if data_set_dropped:
        return f'Data was saved into Mongodb with the name {request_json["collection_name"]}.\n a Collection with the ' \
               f'same name existed, and it was replaced. (lol) '
    return f'Data was saved into Mongodb with the name {request_json["collection_name"]}.'


@app.post('/save_dataset_mongo/force')
async def api_dataset_saving_force(request:Request):
    print(fast_api_print(f'{request.client.host} requested to save the current dataset on MongoDB (Forced)'))
    request_json = await request.json()
    data_to_dict = data.to_dict('records')
    if list(db[request_json['collection_name']].find({})):
        data_set_dropped = True
        db[request_json['collection_name']].drop()
    else:
        data_set_dropped = False

    db[request_json['collection_name']].insert_many(data_to_dict)
    if data_set_dropped:
        return f'Data was saved into Mongodb with the name {request_json["collection_name"]}.\n a Collection with the ' \
               f'same name existed, and it was replaced. (lol) '
    return f'Data was saved into Mongodb with the name {request_json["collection_name"]}.'


@app.post('/save_dataset_mongo/')
async def api_dataset_saving(request:Request):
    print(fast_api_print(f'{request.client.host} requested to save the current dataset on MongoDB'))
    request_json = await request.json()
    data_to_dict = data.to_dict('records')
    if list(db[request_json['collection_name']].find({})):
        return 'A collection with this name already exists, please, choose another. (or use ' \
               '"/save_dataset_mongo/force" to replace.) '
    db[request_json['collection_name']].insert_many(data_to_dict)
    return f'Data was saved into Mongodb with the name {request_json["collection_name"]}.'


@app.post('/prediction/column_equals')
async def api_prediction_columns_equals(request:Request):
    print(fast_api_print(f'{request.client.host} requested to predict based on column and value.'))
    request_json = await request.json()    
    column = request_json['column']
    column_value = request_json['column_value']
    user_id_column = request_json['user_id_column']
    return classify_user(column, column_value, user_id_column)


@app.post('/prediction/all')
async def api_predict_all(request:Request):
    request_json = await request.json()
    print(fast_api_print(f'{request.client.host} requested to predict on all data'))
    model_choice_index = request_json['model_choice_index']
    user_id_column = request_json['user_id_column']
    return classify_all(user_id_column, model_choice_index)


@app.get('/predictions/get_all_made')
async def api_return_predicts(request:Request):
    print(fast_api_print(f'{request.client.host} requested to see all predictions'))
    return PREDICTIONS


@app.get('/save_all_predictions')
async def save_all_predicts_mdb(request:Request):
    print(fast_api_print(f'{request.client.host} requested to save all model info into the current database'))
    db_inserting = {
        "meta_data":{
            "date_of_model_saving":str(datetime.date.today()),
            "time_of_model_saving":datetime.datetime.now().strftime("%H:%M:%S"),
            "user_ip":request.client.host,
        },
        "predictions":PREDICTIONS
    }
    db.predictions.insert_one(db_inserting)
    return {"message":"All predictions were inserted into the current mongodb database under the collection: predictions."}


    
@app.get('/save_model_infos')
async def save_model_infos(request:Request):
    print(fast_api_print(f'{request.client.host} requested to save all model info into the current database'))
    db_inserting = {
        "meta_data":{
            "date_of_model_saving":str(datetime.date.today()),
            "time_of_model_saving":datetime.datetime.now().strftime("%H:%M:%S"),
            "user_ip":request.client.host,
        },
        "model_infos":MODELS_INFO
    }
    db.models_info.insert_one(db_inserting)
    return {"message":"All model info was inserted into the current mongodb database under the collection: models_info."}
