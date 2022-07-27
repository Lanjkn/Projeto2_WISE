import time
import pandas as pd
import numpy as np
from flask import Flask, jsonify, request
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

np.random.seed(1010)

MODELS = []

flask_app = Flask(__name__)

mongo_client = MongoClient('localhost', 27017)

db = mongo_client.projeto_wise

clientes_collection = db.CLIENTES

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


def train_test_spliting(x_param, y_param):
    train_x_, test_x_, train_y_, test_y_ = train_test_split(x_param, y_param, test_size=0.2, stratify=y_param)
    return train_x_, test_x_, train_y_, test_y_


def randomForest_model():
    time_on_creation = time.time()
    print('Creating Random Forest Model...')
    random_forest_classifier = RandomForestClassifier(n_estimators=10)
    print('Fitting data into model for training...')
    random_forest_classifier.fit(train_x, train_y)
    time_to_finish = time.time() - time_on_creation
    print('Done!')

    model_info = {
        'model_type': random_forest_classifier.__class__.__name__,
        'model_parameters': random_forest_classifier.get_params(),
        'seconds_to_create_model': round(time_to_finish, 2),
        'model_scores': scores(random_forest_classifier)
    }

    return {'model_info': model_info, 'model': random_forest_classifier}


def logisticRegression_model():
    time_on_creation = time.time()
    print('Creating Logistic Regression Model...')
    log_regression = LogisticRegression(max_iter=20000)
    print('Fitting data into model for training...')
    log_regression.fit(train_x, train_y)
    time_to_finish = time.time() - time_on_creation
    print('Done!')
    model_info = {
        'model_type': log_regression.__class__.__name__,
        'model_parameters': log_regression.get_params(),
        'seconds_to_create_model': round(time_to_finish, 2),
        'model_scores': scores(log_regression)
    }

    return {'model_info': model_info, 'model': log_regression}


def multilayerPerceptron_model():
    time_on_creation = time.time()
    print('Creating MultiLayer Perceptron Model...')
    mlp_classifier = MLPClassifier()
    print('Fitting data into model for training...')
    mlp_classifier.fit(train_x, train_y)
    time_to_finish = time.time() - time_on_creation
    print('Done!')
    model_info = {
        'model_type': mlp_classifier.__class__.__name__,
        'model_parameters': mlp_classifier.get_params(),
        'seconds_to_create_model': round(time_to_finish, 2),
        'model_scores': scores(mlp_classifier)
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
    model_index = cv_options['model_choice_index']
    try:
        n_splits = cv_options['n_splits']
    except KeyError:
        n_splits = 5
    try:
        shuffle = cv_options['shuffle']
    except KeyError:
        shuffle = True
    try:
        print(f"Doing Cross validation on model {MODELS[model_index].__class__.__name__}")
        validation_results = cross_validate(MODELS[model_index], x, y,
                                            cv=KFold(n_splits=n_splits, shuffle=shuffle))
        return pd.DataFrame(validation_results).to_json()
    except IndexError as e:
        return f'Index out of range.\n you have {len(MODELS)} models created.\n{e}'
    except ValueError as e:
        return f'Invalid Key on JSON file!\n{e}'


def clustering(n_clusters_choice):
    try:
        n_clusters_choice["n_clusters"]
    except KeyError:
        n_clusters_choice["n_clusters"] = 5
    stdscaler = StandardScaler()
    data_scaled = stdscaler.fit_transform(data_dummified)
    try:
        kmeans = KMeans(n_clusters=n_clusters_choice["n_clusters"], n_init=10, max_iter=300, random_state=1010)
        kmeans.fit(data_scaled)
        data_dummified['CLUSTER'] = kmeans.labels_
        description = data_dummified.groupby('CLUSTER')
        n_clients = description.size()
        description = description.mean()
        description['n_clients'] = n_clients
        description_json = description.to_json()
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
    print(
        "Doing the Multilayer Perceptron Classifier Hyper Parameters through Randomized Search Cross Validation... (this might take a while)")
    RSCross_validation = RandomizedSearchCV(MLPClassifier(),
                                            RSCV_parameters, n_iter=10, cv=KFold(n_splits=5, shuffle=True))
    RSCross_validation.fit(train_x, train_y)
    print('Done!')
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
    print(
        "Doing the Logistic Regression Hyper Parameters through Randomized Search Cross Validation... (this might take a while)")
    RSCross_validation = RandomizedSearchCV(LogisticRegression(max_iter=20000),
                                            RSCV_parameters, n_iter=10, cv=KFold(n_splits=5, shuffle=True))
    RSCross_validation.fit(train_x, train_y)
    print('Done!')
    results = pd.DataFrame(RSCross_validation.cv_results_)
    results_html = results.sort_values(by='rank_test_score').to_html()
    results_html_file = open("hyper parameters results - LRC.html", "w")
    results_html_file.write(results_html)
    results_html_file.close()
    print(
        'A HTML file was created with the results dataframe, ordered by their score! (incase you want to review all the models)')
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
    print(
        "Doing the Random Forest Classifier Hyper Parameters through Randomized Search Cross Validation... (this might take a while)")
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


def standardize_data_and_split(x_param, y_param):
    stdscaler = StandardScaler()
    std_x = stdscaler.fit_transform(x_param)
    std_train_x, std_test_x, train_y_, test_y_ = train_test_spliting(std_x, y_param)
    return std_train_x, std_test_x, train_y_, test_y_


def RFC_feature_importance():
    RFC_list = []
    for model_ in MODELS:
        if model_.__class__.__name__ == 'RandomForestClassifier':
            RFC_list.append(model_.__class__.__name__)
    if not RFC_list:
        print('No Random Forest Classifiers created!')
        return
    print(pd.Series(RFC_list))
    rfc_choice = int(input('Which model do you want to see the feature importances?: '))
    feature_importances_series = pd.Series(RFC_list[rfc_choice].feature_importances_,
                                           index=pd.Series([col for col in x.columns]))
    print(feature_importances_series.sort_values(ascending=False) * 100)


@flask_app.route('/')
def home():
    welcome_message = "Welcome to iml api!\nHere, you can a whole lot of stuff, that i will sometime later add" \
                      "\n the api is focused on Machine Learning and model creation!"
    return welcome_message


@flask_app.route('/create_model/RFC')
def api_create_model_RFC():
    model = randomForest_model()
    MODELS.append(model['model'])
    return model['model_info']


@flask_app.route('/create_model/LRC')
def api_create_model_LRC():
    model = logisticRegression_model()
    MODELS.append(model['model'])
    return model['model_info']


@flask_app.route('/create_model/MLP')
def api_create_model_MLP():
    model = multilayerPerceptron_model()
    MODELS.append(model['model'])
    return model['model_info']


@flask_app.route('/model_visualization/')
def api_model_visualization():
    models_dict = {}
    index_test = 0
    for model in MODELS:
        models_dict[str(index_test) + " - " + model.__class__.__name__] = model.get_params()
        index_test += 1
    return jsonify(models_dict)


@flask_app.route('/set_data/', methods=["POST"])
def api_set_data():
    file_loc_json = request.get_json()
    data_dummified = pd.read_csv(file_loc_json['file_loc'])
    x = data_dummified.drop(labels=[file_loc_json['y']], axis=1)
    y = data_dummified[file_loc_json['y']]
    train_x, test_x, train_y, test_y = standardize_data_and_split(x, y)
    return 'new data was set!'


@flask_app.route('/model_score/')
def api_model_scores():
    return jsonify(model_scores())


@flask_app.route('/create_model/hyper_parameters/RFC')
def api_create_model_hp_RFC():
    return hyper_parameters_random_forest()['model_info']


@flask_app.route('/create_model/hyper_parameters/LRC')
def api_create_model_hp_LRC():
    return hyper_parameters_logistic_regression()['model_info']


@flask_app.route('/create_model/hyper_parameters/MLP')
def api_create_model_hp_MLP():
    return hyper_parameters_multilayer_perceptron()['model_info']


@flask_app.route('/cross_validation/', methods=["POST"])
def api_cross_validation():
    cv_options = request.get_json()
    return cross_validation(cv_options)


@flask_app.route('/clustering/', methods=["POST"])
def api_clustering():
    n_clusters = request.get_json()
    return clustering(n_clusters)

@flask_app.route('/predictions/')
def api_prediction():
    prediction = MODELS[0].predict([[data_dummified[data].mean() for data in x.columns]])
    return str(prediction[0])

data_dummified = pd.read_csv('clean_dataset_customer_churn.csv')

x = data_dummified.drop(labels=['SITUACAO'], axis=1)
y = data_dummified['SITUACAO']

train_x, test_x, train_y, test_y = standardize_data_and_split(x, y)

if __name__ == '__main__':
    flask_app.run(debug=True)
