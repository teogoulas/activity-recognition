from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold, train_test_split
import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize
import plotly.graph_objects as go

from utils import constants


def find_activities(data_set):
    activities = []
    for activity in data_set:
        activity_type = activity["activityType"]
        if activity_type not in activities:
            activities.append(activity_type)
    return activities


def group_activities(data_set):
    activity_groups = {"biking": []}
    filtered_data_set = []
    for activity_type in constants.FILTERED_FEATURES:
        if activity_type in ["cycling", "mountain_biking", "trail_running", "other"]:
            continue
        activity_groups[activity_type] = []

    for activity in data_set:
        act_type = activity.activity_type
        if act_type in constants.FILTERED_FEATURES:
            if act_type == "cycling" or act_type == "mountain_biking":
                activity._activity_type = "biking"
                activity_groups["biking"].append(activity)
            elif act_type == "trail_running":
                activity._activity_type = "running"
                activity_groups["running"].append(activity)
            elif act_type == "other":
                activity._activity_type = "indoor_cardio"
                activity_groups["indoor_cardio"].append(activity)
            else:
                activity_groups[act_type].append(activity)
    return activity_groups


def calculate_roc_auc(data_set, optimal_clf, optimal_preprocess_policy, clf_name, splits):
    cv = StratifiedKFold(n_splits=splits)
    X_train_res = data_set[optimal_preprocess_policy]
    y_train_res = data_set["y"]

    tprs = []
    mean_fpr = np.linspace(0, 1, 1000)
    i = 0
    for train, test in cv.split(X_train_res, y_train_res):
        y_train = label_binarize(y_train_res[train], classes=np.unique(y_train_res[train]))
        y_test = label_binarize(y_train_res[test], classes=np.unique(y_train_res[test]))

        classifier = OneVsRestClassifier(optimal_clf.best_estimator_)
        if clf_name in ['knn_classifier', 'naive_bayes', 'decision_tree']:
            y_score = classifier.fit(X_train_res[train], y_train).predict_proba(X_train_res[test])
        else:
            y_score = classifier.fit(X_train_res[train], y_train).decision_function(X_train_res[test])

        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y_test.ravel(), y_score.ravel())
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0

        i += 1

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    return {'auc_score': auc(mean_fpr, mean_tpr), 'fpr': mean_fpr, 'tpr': mean_tpr}


def plot_roc_curve(clf, clf_name, data_set, optimal_preprocess_policy):
    lw = 2

    # Compute micro-average ROC curve and ROC area
    roc = calculate_roc_auc(data_set, clf, optimal_preprocess_policy, clf_name, 10)
    fpr = roc['fpr']
    tpr = roc['tpr']
    roc_auc = roc['auc_score']

    # Plot ROC curve
    data = [go.Scatter(x=fpr, y=tpr,
                       mode='lines',
                       line=dict(color='deeppink', width=lw),
                       name='ROC curve (area = {0:0.2f})'
                            ''.format(roc_auc)),
            go.Scatter(x=[0, 1], y=[0, 1],
                       mode='lines',
                       name='Chance',
                       line=dict(color='black', width=lw, dash='dash'))]

    layout = go.Layout(title=f"ROC curve for optimal {clf_name} classifier",
                       xaxis=dict(title='False Positive Rate'),
                       yaxis=dict(title='True Positive Rate'))

    go.Figure(data=data, layout=layout).show()