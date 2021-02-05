from copy import deepcopy
import numpy as np
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.model_selection import GridSearchCV, cross_val_score, cross_validate
import plotly.graph_objects as go
import warnings

from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')


def classifier_wrapper(clf, grid_params, cv, data_set):
    colors = ['indianred', 'lightsalmon', 'crimson', 'blue', 'green', 'purple', 'goldenrod', 'magenta']
    gs = GridSearchCV(
        clf,
        grid_params,
        verbose=0,
        cv=cv,
        n_jobs=-1,
        return_train_score=True
    )

    # initialize one figure per parameter
    accuracy_figures = {}
    for key in grid_params.keys():
        acc_fig = go.Figure(layout={
            'barmode': 'group',
            'xaxis_tickangle': -45,
            'title': {
                'text': f"Accuracy per {key}",
                'y': 0.9,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            'xaxis_title': key,
            'yaxis_title': "Accuracy Score",
            'legend_title': "Data preprocess algorithm"
        })
        accuracy_figures[key] = acc_fig

    optimal_clf = None
    optimal_preprocess_policy = 'original'

    color_index = 0
    for key in data_set.keys():
        if key == "y":
            continue
        else:
            gs.fit(data_set[key], data_set["y"])

            # find optimal classifier and data scaling algorithm
            if optimal_clf is None or optimal_clf.best_score_ < gs.best_score_:
                optimal_clf = deepcopy(gs)
                optimal_preprocess_policy = key

            for param in grid_params.keys():
                param_test_score = {}
                for value in grid_params[param]:
                    indices = [i for i, x in enumerate(gs.cv_results_[f"param_{param}"]) if x == value]
                    param_test_score[value] = max([gs.cv_results_['mean_test_score'][i] for i in indices])

                acc_fig = accuracy_figures[param]
                acc_fig.add_trace(go.Bar(
                    x=list(map(str, grid_params[param])),
                    y=list(param_test_score.values()),
                    name=key,
                    marker_color=colors[color_index]
                ))

        color_index += 1

    for key in accuracy_figures.keys():
        accuracy_figures[key].show()

    optimal_params = ''
    for key, value in optimal_clf.best_params_.items():
        optimal_params += f"{key}: {value}, "
    optimal_params = optimal_params[:-2]

    return {'optimal_clf': optimal_clf, 'optimal_preprocess_policy': optimal_preprocess_policy,
            'optimal_params': optimal_params}


def logistic_regression(data_set):
    # logistic regression solvers list
    solver_list = ['liblinear', 'newton-cg', 'lbfgs', 'sag', 'saga']
    colors = ['indianred', 'lightsalmon', 'crimson', 'blue', 'green', 'purple']
    parameters = dict(solver=solver_list)

    # evaluation metrics
    scoring = {'AUC': 'roc_auc_ovo_weighted', 'Accuracy': 'accuracy', 'Precision': 'precision_weighted',
               'Recall': 'recall_weighted', 'F1-score': 'f1_weighted'}
    lr = LogisticRegression(random_state=40, multi_class="auto", C=1)
    clf = GridSearchCV(lr, parameters, cv=10, scoring=scoring, refit='Accuracy', return_train_score=True)

    # initialize one figure per Metric
    figures = {}
    for key in scoring.keys():
        fig = go.Figure(layout={
            'barmode': 'group',
            'xaxis_tickangle': -45,
            'title': {
                'text': f"{key} Metric per Logistic regression solver",
                'y': 0.9,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            'xaxis_title': "Solver",
            'yaxis_title': f"{key} Score",
            'legend_title': "Data preprocess algorithm"
        })
        figures[key] = fig

    color_index = 0
    optimal_clf = None
    optimal_preprocess_policy = 'original'
    for key in data_set.keys():
        if key == "y":
            continue
        else:
            solver_score = {}
            clf.fit(data_set[key], data_set["y"])

            # find optimal classifier and data scaling algorithm
            if optimal_clf is None or optimal_clf.best_score_ < clf.best_score_:
                optimal_clf = deepcopy(clf)
                optimal_preprocess_policy = key

            for score_fn in scoring.keys():
                scores = clf.cv_results_[f"mean_test_{score_fn}"]

                for score, solver, in zip(scores, solver_list):
                    solver_score[solver] = score

                fig = figures[score_fn]
                fig.add_trace(go.Bar(
                    x=solver_list,
                    y=list(solver_score.values()),
                    name=key,
                    marker_color=colors[color_index]
                ))

            color_index += 1

    for key in figures.keys():
        figures[key].show()

    return {'optimal_clf': optimal_clf, 'optimal_preprocess_policy': optimal_preprocess_policy}


def knn_classifier(data_set):
    grid_params = {
        'n_neighbors': [1, 3, 5, 11, 19],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }
    return classifier_wrapper(KNeighborsClassifier(), grid_params, 10, data_set)


def naive_bayes(data_set):
    colors = ['indianred', 'lightsalmon', 'crimson', 'blue', 'green', 'purple']

    # evaluation metrics
    scoring = {'AUC': 'roc_auc_ovo_weighted', 'Accuracy': 'accuracy', 'Precision': 'precision_weighted',
               'Recall': 'recall_weighted', 'F1-score': 'f1_weighted'}

    # initialize one figure per Metric

    fig = go.Figure(layout={
        'barmode': 'group',
        'xaxis_tickangle': -45,
        'title': {
            'text': f"Naive Bayes Model evaluation Metrics",
            'y': 0.9,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        'xaxis_title': "Metrics",
        'yaxis_title': f"Score",
        'legend_title': "Data preprocess algorithm"
    })

    color_index = 0
    optimal_clf = None
    optimal_preprocess_policy = 'original'
    for key in data_set.keys():
        if key == "y":
            continue
        else:
            scores = {}
            for score_fn in scoring.keys():
                cv = cross_validate(GaussianNB(), data_set[key], data_set["y"], return_train_score=True,
                                    scoring=scoring[score_fn], cv=10)
                scores[score_fn] = np.mean(cv['test_score'])

                if score_fn == 'Accuracy' and (optimal_clf is None or optimal_clf['best_score_'] < scores[score_fn]):
                    optimal_clf = {
                        'best_score_': scores[score_fn],
                        'preprocess_policy': key,
                        'fit_time_': np.mean(cv['fit_time'])
                    }
                    optimal_preprocess_policy = key

            fig.add_trace(go.Bar(
                x=list(scoring.keys()),
                y=list(scores.values()),
                name=key,
                marker_color=colors[color_index]
            ))

            color_index += 1

    fig.show()

    return {'optimal_clf': optimal_clf, 'optimal_preprocess_policy': optimal_preprocess_policy}


def decision_tree(data_set):
    grid_params = {
        'criterion': ['gini', 'entropy'],
        'splitter': ['best', 'random'],
        'max_depth': [None, 1, 5, 10, 15, 20, 25]
    }
    return classifier_wrapper(DecisionTreeClassifier(random_state=45), grid_params, 10, data_set)


def svm_classifier(data_set):
    grid_params = {
        'C': [0.1, 1, 10, 100],
        'gamma': [1, 0.1, 0.01, 0.001],
        'kernel': ['rbf', 'poly', 'sigmoid']
    }
    return classifier_wrapper(SVC(random_state=45), grid_params, 10, data_set)


def perceptron(data_set):
    grid_params = {
        'eta0': [0.0001, 0.001, 0.01, 0.1, 1.0],
        'max_iter': [100, 1000, 10000]
    }
    return classifier_wrapper(Perceptron(random_state=45), grid_params, 10, data_set)
