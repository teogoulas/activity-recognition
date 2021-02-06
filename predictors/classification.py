import sys
import pandas as pd
from predictors.predictors import logistic_regression, knn_classifier, naive_bayes, decision_tree, svm_classifier, \
    perceptron, comparing_classifiers
from utils import data_handler as json
from utils.data_handler import data_preprocessing


def classification(args):
    try:
        generate_raw_data = bool(args.generate_raw_data)
    except ValueError:
        print("Provide an option!")
        sys.exit()

    if generate_raw_data:
        json_object = json.import_json()
        raw_data_set = json.extract_features(json_object)
    else:
        raw_data_set = pd.read_csv(r'/meta_data/raw_data.csv')

    data_set = data_preprocessing(raw_data_set)

    lr = logistic_regression(data_set)
    print(f"Optimal logistic regression classifier solver: {lr['optimal_clf'].best_params_['solver']}")
    print(f"with optimal data preprocessing policy: {lr['optimal_preprocess_policy']}")
    print(f"Accuracy score: {lr['optimal_clf'].best_score_}")
    print()

    knn = knn_classifier(data_set)
    print(f"Optimal KNN classifier params: {knn['optimal_params']}")
    print(f"with optimal data preprocessing policy: {knn['optimal_preprocess_policy']}")
    print(f"Accuracy score: {knn['optimal_clf'].best_score_}")
    print()

    nbc = naive_bayes(data_set)
    print(f"Optimal Naive Bayes accuracy score: {nbc['optimal_clf'].best_score_['Accuracy']}")
    print(f"with optimal data preprocessing policy: {nbc['optimal_preprocess_policy']}")
    print()

    dt = decision_tree(data_set)
    print(f"Optimal Decision tree classifier params: {dt['optimal_params']}")
    print(f"with optimal data preprocessing policy: {dt['optimal_preprocess_policy']}")
    print(f"Accuracy score: {dt['optimal_clf'].best_score_}")
    print()

    svm = svm_classifier(data_set)
    print(f"Optimal SVM classifier params: {svm['optimal_params']}")
    print(f"with optimal data preprocessing policy: {svm['optimal_preprocess_policy']}")
    print(f"Accuracy score: {svm['optimal_clf'].best_score_}")
    print()

    prc = perceptron(data_set)
    print(f"Optimal Perceptron params: {prc['optimal_params']}")
    print(f"with optimal data preprocessing policy: {prc['optimal_preprocess_policy']}")
    print(f"Accuracy score: {prc['optimal_clf'].best_score_}")
    print()

    clfs = {
        'logistic_regression': lr,
        'knn_classifier': knn,
        'naive_bayes': nbc,
        'decision_tree': dt,
        'svm_classifier': svm,
        'perceptron': prc
    }
    comparing_classifiers(clfs, data_set)
