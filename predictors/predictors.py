from copy import deepcopy

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import plotly.graph_objects as go
import warnings

warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')


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