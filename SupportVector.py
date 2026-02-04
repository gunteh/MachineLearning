from sklearn.svm import SVC, SVR

class SVMModel:
    def __init__(self, model_type="classifier", **kwargs):
        """
        Class for a Support Vector Machine (SVM) classifier or regressor.

        Parameters:
        - model_type: "classifier" or "regressor"
        - kwargs: hyperparameters passed directly to SVC or SVR
        """
        self.model_type = model_type.lower()
        self.params = self._set_defaults(kwargs)

        if self.model_type == "classifier":
            self.model = SVC(**self.params)
        elif self.model_type == "regressor":
            self.model = SVR(**self.params)
        else:
            raise ValueError("model_type must be 'classifier' or 'regressor'")

    def _set_defaults(self, custom_params):
        """Set default parameters, overridden by user-provided values."""
        defaults = {
            "kernel": "rbf",        # Options: 'linear', 'poly', 'rbf', 'sigmoid'
            "C": 1.0,               # For regularization strength
            "gamma": "scale",       # Kernel coefficient for use with gamma
            "degree": 3,            # Only necessary for 'poly' kernel
            "probability": True if self.model_type == "classifier" else False, #logic to prevent the default from erroring out
            "shrinking": True,
            "tol": 1e-3,
            "max_iter": -1,
        }
        return {**defaults, **custom_params}

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        """Predicting class probabilities (classifier use only, and ONLY if probability=True)."""
        if self.model_type != "classifier":
            raise AttributeError("predict_proba is only available for classifiers")
        if not getattr(self.model, "probability", False):
            raise AttributeError("SVM was not composed with probability=True")
        return self.model.predict_proba(X)

    def score(self, X, y):
        """Returns either accuracy (classifier) or R^2 score (regressor)."""
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
    
    def decision_function(self, X):
        return self.model.decision_function(X)