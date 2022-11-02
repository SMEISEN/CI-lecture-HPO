import warnings
import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings(action='ignore')  # some configurations will not converge


if __name__ == '__main__':

    # load dataset
    dataset = load_diabetes()
    X = dataset["data"]
    y = dataset["target"]

    # create machine learning pipeline
    pipe = make_pipeline(
        MinMaxScaler(),
        MLPRegressor()
    )

    # define hyperparameters to be assessed
    params = {    # 756 evaluations
        "mlpregressor__hidden_layer_sizes": [(i,) for i in range(3, 31)] +
                                            [(i, i,) for i in range(3, 31)] +
                                            [(i, i, i,) for i in range(3, 31)],
        "mlpregressor__activation": ["logistic", "tanh", "relu"],
        "mlpregressor__solver": ["lbfgs", "sgd", "adam"]
    }

    results = {"gs": {"best_params": [], "val_score": [], "test_score": []},
               "rs": {"best_params": [], "val_score": [], "test_score": []}}
    for i in range(10):
        # split data into training and test dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

        print(f"Start random search")
        # assess parameters with 5-fold cross-validation
        rs = RandomizedSearchCV(pipe, params, cv=5, n_iter=250, scoring="neg_mean_absolute_error", refit=True)
        rs.fit(X_train, y_train)
        results["rs"]["best_params"].append(rs.best_params_)
        results["rs"]["val_score"].append(rs.best_score_)
        results["rs"]["test_score"].append(mean_absolute_error(y_test, rs.predict(X_test)))

        print(f"Start grid search")
        # assess parameters with 5-fold cross-validation
        gs = GridSearchCV(pipe, params, cv=5, scoring="neg_mean_absolute_error", refit=True)
        gs.fit(X_train, y_train)
        results["gs"]["best_params"].append(gs.best_params_)
        results["gs"]["val_score"].append(gs.best_score_)
        results["gs"]["test_score"].append(mean_absolute_error(y_test, gs.predict(X_test)))

        print(f"Finished round {i + 1}/10")

    gs_mean = np.mean(results["gs"]["test_score"])
    gs_std = np.std(results["gs"]["test_score"])
    rs_mean = np.mean(results["rs"]["test_score"])
    rs_std = np.std(results["rs"]["test_score"])

    search_methods = ["Rastersuche", "Monte Carlo Methode"]
    x_pos = np.arange(len(search_methods))
    CTEs = [gs_mean, rs_mean]
    error = [gs_std, rs_std]

    fig, ax = plt.subplots()
    ax.bar(x_pos, CTEs, yerr=error, align='center', alpha=0.5, ecolor='black', capsize=10)
    ax.set_ylabel("RMSE on test split")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(search_methods)
    ax.yaxis.grid(True)

    plt.show()

    fig.savefig("gs756_v_rs756-250.jpg")
