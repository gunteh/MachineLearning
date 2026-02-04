from sklearn.neural_network import MLPClassifier, MLPRegressor

class NNModel:
    def __init__(self, model_type="classifier", **kwargs):
        """
        Modeling for a Neural Network classifier or regressor.

        Parameters:
        - model_type: "classifier" or "regressor"
        - kwargs: any hyperparameters for MLPClassifier or MLPRegressor
        """
        self.model_type = model_type.lower()
        self.params = self._set_defaults(kwargs)

        if self.model_type == "classifier":
            self.model = MLPClassifier(**self.params)
        elif self.model_type == "regressor":
            self.model = MLPRegressor(**self.params)
        else:
            raise ValueError("model_type must be 'classifier' or 'regressor', please try again")

    def _set_defaults(self, custom_params):
        """Default parameters for MLP, overridden by custom ones."""
        defaults = {
            "hidden_layer_sizes": (100,),  # option; single hidden layer with 100 neurons
            "activation": "relu",          # options: 'identity', 'logistic', 'tanh', 'relu'
            "solver": "adam",              # options: 'lbfgs', 'sgd', 'adam'
            "alpha": 0.0001,               # options: L2 penalty (regularization)
            "batch_size": 'auto',
            "learning_rate": "constant",   # options: 'constant', 'invscaling', 'adaptive'
            "max_iter": 200,
            "random_state": 42,
            "early_stopping": False,
        }
        return {**defaults, **custom_params}

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        """ONLY available for classification."""
        if self.model_type != "classifier":
            raise AttributeError("predict_proba is only available for classifiers")
        return self.model.predict_proba(X)

    def score(self, X, y):
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
        return self.params

    def predict_proba(self, X):
        return self.model.predict_proba(X)