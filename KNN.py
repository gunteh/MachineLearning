from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

class KNNModel:
    def __init__(self, model_type="classifier", **kwargs):
        """
        Class for K-Nearest Neighbors (KNN) classifier or regressor generator

        Parameters set:
        - model_type: "classifier" or "regressor"
        - kwargs: model paaramters, passed directly to the sklearn model
        """
        self.model_type = model_type.lower()
        self.params = self._set_defaults(kwargs)

        if self.model_type == "classifier":
            self.model = KNeighborsClassifier(**self.params)
        elif self.model_type == "regressor":
            self.model = KNeighborsRegressor(**self.params)
        else:
            raise ValueError("model_type must be 'classifier' or 'regressor', please try aagain")

    def _set_defaults(self, custom_params):
        """Default KNN parameters, overridden by custom values as needed."""
        defaults = {
            "n_neighbors": 5,
            "weights": "uniform",     # options: 'uniform' or 'distance' weights
            "algorithm": "auto",      # options; 'auto', 'ball_tree', 'kd_tree', 'brute'
            "leaf_size": 30,
            "p": 2,                   # options: 1 = Manhattan, 2 = Euclidean
            "metric": "minkowski",
            "n_jobs": None,            # option: Use all CPUs if -1
        }
        return {**defaults, **custom_params}

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        """Predict class probabilities (ONLY classifier use)."""
        if self.model_type != "classifier":
            raise AttributeError("predict_proba is only available for classifiers")
        return self.model.predict_proba(X)

    def score(self, X, y):
        """Returning the R^2 score for regression or accuracy for classification."""
        return self.model.score(X, y)
    

    def get_params(self, deep=True):
        #sourced from https://scikit-learn.org/stable/developers/develop.html#get-params-and-set-params
        return self.params.copy()

    def set_params(self, **params):
        # sourced from #https://scikit-learn.org/stable/developers/develop.html#get-params-and-set-params
        self.params.update(params)
        if self.estimator_type == "classifier":
            self.model = DecisionTreeClassifier(**self.params)
        else:
            self.model = DecisionTreeRegressor(**self.params)
        return self

    def params_inuse(self):
        """Returning the hyperparameters currently being used."""
        return self.params

    def predict_proba(self, X):
        return self.model.predict_proba(X)