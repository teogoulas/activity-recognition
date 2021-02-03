from predictors.predictors import logistic_regression, knn_classifier
from utils import data_handler as json
from utils import utilities as utils
import plotly.express as px

if __name__ == '__main__':
    json_object = json.import_json()
    data_set = json.extract_features(json_object)
    lr = logistic_regression(data_set)
    print(f"Optimal logistic regression classifier solver: {lr['optimal_clf'].best_params_['solver']}")
    print(f"with optimal data preprocessing policy: {lr['optimal_preprocess_policy']}")
    print(f"Accuracy score: {lr['optimal_clf'].best_score_}")

    knn = knn_classifier(data_set)
    print(f"Optimal KNN classifier params: {knn['optimal_params']}")
    print(f"with optimal data preprocessing policy: {knn['optimal_preprocess_policy']}")
    print(f"Accuracy score: {knn['optimal_clf'].best_score_}")
