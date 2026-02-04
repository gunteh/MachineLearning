import pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

class DecisionTreeModel:
    def __init__(self, model_type="classifier", **kwargs):
        """
        Model for Decision Tree

        Parameters (and usage reasons):
        - basic_type: "classifier" or "regressor". For simplicity and felxibility
        - kwargs: hyperparameters for the underlying model, customize to hearts content
        """
        self.estimator_type = model_type
        self.params = self._set_defaults(kwargs)

        if model_type == "classifier":
            self.model = DecisionTreeClassifier(**self.params)
        elif model_type == "regressor":
            self.model = DecisionTreeRegressor(**self.params)
        else:
            raise ValueError("model_type must be 'classifier' or 'regressor'. Please try again")

    def _set_defaults(self, custom_params):
        """This is only for the first base case when paramters are not customized."""
        defaults = {
            "criterion": "gini" if self.estimator_type == "classifier" else "squared_error",
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "random_state": 42,
        }
        return {**defaults, **custom_params}

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

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
        """Show all the parameters thata are being or are in use."""
        return self.params

    def predict_proba(self, X):
        if self.estimator_type == "classifier":
            return self.model.predict_proba(X)

