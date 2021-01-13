import numpy as np


def train_models(groups):
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    for activity_group in groups:
        train_range = round(0.8*len(groups))
        x_train.append(activity_group[1:train_range, :])
        y_train.append(np)
        y_train.append(activity_group[train_range:len(groups), :])
