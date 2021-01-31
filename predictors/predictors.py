import numpy as np
from sklearn import model_selection as ms
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def data_preprocessing(data_set):
    # remove NaN value and convert to np array
    filtered_data = data_set.dropna()
    y = filtered_data['activity_type'].to_numpy()
    x = filtered_data.iloc[:, 1:11]
    x.boxplot(column=list(x.columns))
    x = x.to_numpy()
    X_train, X_test, y_train, y_test = ms.train_test_split(
        x, y, stratify=y, test_size=0.2)
    return {"x_train": X_train, "x_test": X_test, "y_train": y_train, "y_test": y_test}


def logistic_regression(data_set):
    models = data_preprocessing(data_set)
    log_reg = LogisticRegression()
    params = dict(solver=['lbfgs'])
    pipe = make_pipeline(StandardScaler(), GridSearchCV(log_reg, params, cv=10))
    pipe.fit(models["x_train"], models["y_train"])
    score = pipe.score(models["x_test"], models["y_test"])
    print(score)

