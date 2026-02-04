import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from functools import partial

from DecisionTree import DecisionTreeModel
from KNN import KNNModel
from NN import NNModel
from SupportVector import SVMModel
from Test_Train import *
from DataProcessing import *

def main():
    print("Starting main.py")
    us_accidents = pd.read_csv("US_Accidents_March23.csv")
    print(us_accidents.head())

    #loading dataexit
    hotel_data = pd.read_csv("hotel_bookings.csv")
    print(hotel_data.head())

    # setting taarget
    X = hotel_data.drop(columns=["is_canceled"])
    y = hotel_data["is_canceled"]
    X_2 = us_accidents.drop(columns=["Severity"])
    y_2 = us_accidents["Severity"]

    print("Hotel Input shape:", X.shape)
    print("Hotel Target shape:", y.shape)
    print("Hotel Number of input features:", X.shape[1])
    print("Hotel Number of output classes:", len(set(y)))



    # split into test train samples
    X_unique = X.drop_duplicates()
    y_unique = y.loc[X_unique.index]
    X_accidents_unique = X_2.drop_duplicates()
    y_accidents_unique = y_2.loc[X_accidents_unique.index]

    max_samples_amount = 1000000
    if len(X_accidents_unique) > max_samples_amount:
        X_accidents_unique, y_accidents_unique = resample(
            X_accidents_unique, y_accidents_unique,
            replace=False,
            n_samples=max_samples_amount,
            random_state=123
        )

    print("Us_accdients Input shape:", X_accidents_unique.shape)
    print("Us_accdients Target shape:", y_accidents_unique.shape)
    print("Us_accdients Number of input features:", X_accidents_unique.shape[1])
    print("Us_accdients Number of output classes:", len(set(y_accidents_unique)))

    # resetting index, just in case
    X_unique = X_unique.reset_index(drop=True)
    y_unique = y_unique.reset_index(drop=True)
    X_accidents_unique = X_accidents_unique.reset_index(drop=True)
    y_accidents_unique = y_accidents_unique.reset_index(drop=True)

    # Now split unique data
    X_train, X_test, y_train, y_test = train_test_split(
        X_unique, y_unique, test_size=0.2, random_state=123)

    X_accidents_train, X_accidents_test, y_accidents_train, y_accidents_test = train_test_split(
        X_accidents_unique, y_accidents_unique, test_size=0.2, random_state=123, stratify=y_accidents_unique)

    # hotel dataset columns for encoding/scaling
    numeric_cols_hotel = X_train.select_dtypes(include=['number']).columns.tolist()
    categorical_cols_hotel = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

    # accidents dataset columns for encoding/scaling
    numeric_cols_accidents = X_accidents_train.select_dtypes(include=['number']).columns.tolist()
    categorical_cols_accidents = X_accidents_train.select_dtypes(include=['object', 'category']).columns.tolist()


    print("Train target distribution:\n", y_train.value_counts(normalize=True))
    print("Test target distribution:\n", y_test.value_counts(normalize=True))

    """   # X_train_processed = data_processing_full(X_train) #using the full model created a lot of calculation time for the SVM, and NN did not converge
    X_train_processed, top_agents, top_companies, top_countries = data_processing_lite(X_train)
    # X_test_processed = data_processing_full(X_test) #
    X_test_processed, _, _, _ = data_processing_lite(X_test, top_agents, top_companies, top_countries)

    # create one hot encoding:
    X_train_processed = pd.get_dummies(X_train_processed, drop_first=True)
    X_test_processed = pd.get_dummies(X_test_processed, drop_first=True)

    # align columns as needed
    X_test_processed = X_test_processed.reindex(columns=X_train_processed.columns, fill_value=0)

    #scale data for KNN, NN, and SVM
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_processed)  # Fit only on training data
    X_test_scaled = scaler.transform(X_test_processed)"""

    # build a model list

    modelsLearningCurveOG = [{
        "name": "DecisionTree_5Max",
        "model": DecisionTreeModel(model_type="classifier", max_depth=5),
        "scaled": False,
        "version": "LearningCurve"
        },
        {
        "name": "KNN_5",
        "model": KNNModel(model_type="classifier",  
            n_neighbors=5,
            weights="distance",
            algorithm="auto",
            metric="euclidean"
        ),
        "scaled": True,
        "version": "LearningCurve"
        },
                {"name": "NeuralNet_basicLayers",
        "model": NNModel(model_type="classifier",
            hidden_layer_sizes=(64,),
            activation="relu",
            solver="sgd",
            learning_rate_init=0.01,
            batch_size=32,
            max_iter=200,
            alpha=0.0001,
            early_stopping=True
        ),
        "scaled": True,
        "version": "LearningCurve"
        },]

    modelsLearningCurve = [
    
        {
        "name": "SVM_rbf",
        "model": SVMModel(model_type="classifier", kernel="rbf", C=1.0, gamma=0.1),
        "scaled": True,
        "version": "LearningCurve"
        },
    ]

    modelstesting = [
        {
        "name": "DecisionTree_5Max",
        "model": DecisionTreeModel(model_type="classifier", max_depth=5),
        "scaled": False,
        "version": "Testing"
        },
        {
        "name": "DecisionTree_10Max",
        "model": DecisionTreeModel(model_type="classifier", max_depth=10),
        "scaled": False,
        "version": "Testing"
        },
        {
        "name": "DecisionTree_20Max",
        "model": DecisionTreeModel(model_type="classifier", max_depth=20),
        "scaled": False,
        "version": "Testing"
        },
        {
        "name": "KNN_5",
        "model": KNNModel(model_type="classifier",  
            n_neighbors=5,
            weights="distance",
            algorithm="auto",
            metric="euclidean"
        ),
        "scaled": True,
        "version": "Testing"
        },
        {
        "name": "KNN_10",
        "model": KNNModel(model_type="classifier",  
            n_neighbors=10,
            weights="distance",
            algorithm="auto",
            metric="euclidean"
        ),
        "scaled": True,
        "version": "Testing"
        },
        {
        "name": "KNN_15",
        "model": KNNModel(model_type="classifier",  
            n_neighbors=15,
            weights="distance",
            algorithm="auto",
            metric="euclidean"
        ),
        "scaled": True,
        "version": "Testing"
        },
        {
        "name": "NeuralNet_basicLayers",
        "model": NNModel(model_type="classifier",
            hidden_layer_sizes=(128,),
            activation="relu",
            solver="sgd",
            learning_rate_init=0.01,
            batch_size=32,
            max_iter=200,
            alpha=0.0001,
            early_stopping=True
        ),
        "scaled": True,
        "version": "Testing"
        },
        {
        "name": "NeuralNet_MediumLayers",
        "model": NNModel(model_type="classifier",
            hidden_layer_sizes=(64, 64),
            activation="relu",
            solver="sgd",
            learning_rate_init=0.01,
            batch_size=32,
            max_iter=200,
            alpha=0.0001,
            early_stopping=True
        ),
        "scaled": True,
        "version": "Testing"
        },
        {
        "name": "NeuralNet_LargeLayers",
        "model": NNModel(model_type="classifier",
            hidden_layer_sizes=(32, 32, 32, 32),
            activation="relu",
            solver="sgd",
            learning_rate_init=0.01,
            batch_size=32,
            max_iter=200,
            alpha=0.0001,
            early_stopping=True
        ),
        "scaled": True,
        "version": "Testing"
        },
        {
        "name": "SVM_rbf",
        "model": SVMModel(model_type="classifier", kernel="rbf", C=1.0, gamma=0.1),
        "scaled": True,
        "version": "Testing"
        },
        {
        "name": "SVM_linear",
        "model": SVMModel(model_type="classifier", kernel="linear", C=1.0),
        "scaled": True,
        "version": "Testing"
        }, 
        {
        "name": "SVM_poly",
        "model": SVMModel(model_type="classifier", kernel="poly", degree=3, C=1.0, gamma=0.1),
        "scaled": True,
        "version": "Testing"
        }
    ]

    modelsValidation = [
        {
        "name": "DecisionTree_10Max",
        "model": DecisionTreeModel(model_type="classifier", max_depth=10),
        "scaled": False,
        "version": "Validation"
        },
    ]

    """make_hotel_scaled = partial(make_hotel_scaled_pipeline, 
                                numeric_cols=numeric_cols_hotel, 
                                categorical_cols=categorical_cols_hotel)

    make_hotel_unscaled = partial(make_hotel_unscaled_pipeline, 
                            numeric_cols=numeric_cols_hotel, 
                            categorical_cols=categorical_cols_hotel)

    make_accidents_scaled = partial(make_accidents_scaled_pipeline, 
                                numeric_cols=numeric_cols_accidents, 
                                categorical_cols=categorical_cols_accidents)

    make_accidents_unscaled = partial(make_accidents_unscaled_pipeline, 
                                numeric_cols=numeric_cols_accidents, 
                                categorical_cols=categorical_cols_accidents)"""

    """run_model_set(
        modelsLearningCurve,
        "hotel",
        X_train, X_test, y_train, y_test,
        make_hotel_scaled,
        make_hotel_unscaled
    )"""

    run_model_set(
        modelsLearningCurve,
        "accidents",
        X_accidents_train, X_accidents_test, y_accidents_train, y_accidents_test,
        make_accidents_scaled_pipeline,
        make_accidents_unscaled_pipeline, X_sample = X_accidents_train
    )

    run_model_set(
        modelstesting,
        "hotel",
        X_train, X_test, y_train, y_test,
        make_hotel_scaled_pipeline,
        make_hotel_unscaled_pipeline, X_sample = X_train
    )

    run_model_set(
        modelstesting,
        "accidents",
        X_accidents_train, X_accidents_test, y_accidents_train, y_accidents_test,
        make_accidents_scaled_pipeline,
        make_accidents_unscaled_pipeline, X_sample = X_accidents_train
    )
    """    results = []
    dataset = "hotel"
    for item in modelsLearningCurve:
        model = item["model"]
        version = item["version"]
        name = item["name"]
        use_scaled = item["scaled"]

        pipeline = make_hotel_scaled_pipeline(model) if use_scaled else make_hotel_unscaled_pipeline(model)


        plot_learning_curve(pipeline, X_train_input, y_train, name, version, dataset)

        print(f"[{dataset}] Running {name} model with {'scaled' if use_scaled else 'unscaled'} data")
        result = run_model(model, X_train_input, y_train, X_test_input, y_test, name, version, dataset)
        results.append((dataset, name, version, result))

    for item in modelstesting:
        model = item["model"]
        version = item["version"]
        name = item["name"]
        use_scaled = item["scaled"]

        pipeline = make_hotel_scaled_pipeline(model) if use_scaled else make_hotel_unscaled_pipeline(model)

        print(f"[{dataset}] Running {name} model with {'scaled' if use_scaled else 'unscaled'} data")
        pipeline.fit(X_train, y_train)
        result = run_model(pipeline, X_train, y_train, X_test, y_test, item["name"], item["version"])
        results.append((dataset, name, version, result))
    
    dataset = "accidents"
    for item in modelstesting:
        model = item["model"]
        version = item["version"]
        name = item["name"]
        use_scaled = item["scaled"]

        pipeline = make_accidents_scaled_pipeline(model) if use_scaled else make_accidents_unscaled_pipeline(model)

        print(f"[{dataset}] Running {name} model with {'scaled' if use_scaled else 'unscaled'} data")
        pipeline.fit(X_train, y_train)
        result = run_model(pipeline, X_train, y_train, X_test, y_test, item["name"], item["version"])
        results.append((dataset, name, version, result))"""

    """for item in modelsValidation:
        model = item["model"]
        version = item["version"]
        name = item["name"]
        use_scaled = item["scaled"]

        X_train_input = X_train_scaled if use_scaled else X_train_processed
        X_test_input = X_test_scaled if use_scaled else X_test_processed

        print(f"Running {name} model with {'scaled' if use_scaled else 'unscaled'} data")
        result = run_model(model, X_train_input, y_train, X_test_input, y_test, name, version)
        results.append((name, version, result))"""


if __name__ == "__main__":
    main()