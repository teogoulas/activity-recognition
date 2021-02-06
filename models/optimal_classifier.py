class OptimalClassifier:

    def __init__(self, best_estimator, best_score, preprocess_policy, fit_time):
        self.best_estimator_ = best_estimator
        self.best_score_ = best_score
        self.preprocess_policy_ = preprocess_policy
        self.fit_time_ = fit_time
