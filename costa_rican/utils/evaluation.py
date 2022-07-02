from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import RFECV

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
from IPython.display import display


def make_custom_scorer():
    """
    custom scorer를 반환한다.
    """
    return make_scorer(f1_score, greater_is_better=True, average="macro")


class CVScore:
    def __init__(self, clf, seed=2022, model_params=None, scorer_params=None, cv=10):
        model_params["random_state"] = seed
        self.clf = clf(**model_params)
        self.scorer = make_scorer(**scorer_params)
        self.cv = cv

    def train(self, x_train, y_train):
        cv_score = cross_val_score(self.clf, x_train, y_train, cv=self.cv, scoring=self.scorer)
        print(f"{str(self.clf.__class__).rsplit('.', 1)[-1][:-2]}")
        print(
            f"10 Fold Cross Validation F1 Score = {round(cv_score.mean(), 4)} with std = {round(cv_score.std(), 4)}"
        )


class RFEWithCV:
    def __init__(self, model, scorer, n_cv=3, step=1, n_jobs=-1):
        self.selector = RFECV(model, step=step, cv=n_cv, scoring=scorer, n_jobs=n_jobs)

    def __call__(self, train_set, train_labels):
        self.selector.fit(train_set, train_labels)
        plt.plot(self.selector.grid_scores_.mean(axis=1))
        plt.xlabel("Number of Features")
        plt.ylabel("Macro F1 Score")
        plt.title("Feature Selection Scores")
        print("n_features")
        print(f"{self.selector.n_features_}")
        return self.selector


def macro_f1_score(labels, predictions):
    """
    early stopping을 적용하기 위한 custom evaluation function
    """
    # Reshape the predictions as needed
    predictions = predictions.reshape(len(np.unique(labels)), -1).argmax(axis=0)

    metric_value = f1_score(labels, predictions, average="macro")

    # Return is name, value, is_higher_better
    return "macro_f1", metric_value, True


def model_gbm(
    features,
    labels,
    test_features,
    test_ids,
    submission_base,
    nfolds=5,
    return_preds=False,
    hyp=None,
):
    """Model using the GBM and cross validation.
    Trains with early stopping on each fold.
    Hyperparameters probably need to be tuned."""

    feature_names = list(features.columns)

    if hyp is not None:
        # Using early stopping so do not need number of esimators
        if "n_estimators" in hyp:
            del hyp["n_estimators"]
        params = hyp

    else:
        # Model hyperparameters
        params = {
            "boosting_type": "dart",
            "colsample_bytree": 0.88,
            "learning_rate": 0.028,
            "min_child_samples": 10,
            "num_leaves": 36,
            "reg_alpha": 0.76,
            "reg_lambda": 0.43,
            "subsample_for_bin": 40000,
            "subsample": 0.54,
            "class_weight": "balanced",
            }

    model = lgb.LGBMClassifier(
        **params, objective="multiclass", n_jobs=-1, n_estimators=10000, random_state=10
    )

    strkfold = StratifiedKFold(n_splits=nfolds, shuffle=True)

    predictions = pd.DataFrame()
    importances = np.zeros(len(feature_names))
    features = np.array(features)
    test_features = np.array(test_features)
    labels = np.array(labels).reshape((-1))
    valid_scores = []
    

    for i, (train_indices, valid_indices) in enumerate(strkfold.split(features, labels)):

        # Dataframe for fold predictions
        fold_predictions = pd.DataFrame()

        # Training and validation data
        X_train = features[train_indices]
        X_valid = features[valid_indices]
        y_train = labels[train_indices]
        y_valid = labels[valid_indices]

        # Train with early stopping
        model.fit(
            X_train,
            y_train,
             early_stopping_rounds = 100,
            eval_metric=macro_f1_score,
            eval_set=[(X_train, y_train), (X_valid, y_valid)],
            eval_names=["train", "valid"],
            verbose=200,
        )

        valid_scores.append(model.best_score_["valid"]["macro_f1"])

        # Make predictions from the fold as probabilities
        fold_probabilitites = model.predict_proba(test_features)

        # Record each prediction for each class as a separate column
        for j in range(4):
            fold_predictions[(j + 1)] = fold_probabilitites[:, j]

        # Add needed information for predictions
        fold_predictions["idhogar"] = test_ids
        fold_predictions["fold"] = i + 1

        # Add the predictions as new rows to the existing predictions
        predictions = predictions.append(fold_predictions)

        # Feature importances
        importances += model.feature_importances_ / nfolds

        # Display fold information
        display(
            f"Fold {i + 1}, Validation Score: {round(valid_scores[i], 5)}, Estimators Trained: {model.best_iteration_}"
        )

    # Feature importances dataframe
    feature_importances = pd.DataFrame({"feature": feature_names, "importance": importances})

    valid_scores = np.array(valid_scores)
    display(
        f"{nfolds} cross validation score: {round(valid_scores.mean(), 5)} with std: {round(valid_scores.std(), 5)}."
    )

    # If we want to examine predictions don't average over folds
    if return_preds:
        predictions["Target"] = predictions[[1, 2, 3, 4]].idxmax(axis=1)
        predictions["confidence"] = predictions[[1, 2, 3, 4]].max(axis=1)
        return predictions, feature_importances

    # Average the predictions over folds
    predictions = predictions.groupby("idhogar", as_index=False).mean()

    # Find the class and associated probability
    predictions["Target"] = predictions[[1, 2, 3, 4]].idxmax(axis=1)
    predictions["confidence"] = predictions[[1, 2, 3, 4]].max(axis=1)
    predictions = predictions.drop(columns=["fold"])

    # Merge with the base to have one prediction for each individual
    submission = submission_base.merge(
        predictions[["idhogar", "Target"]], on="idhogar", how="left"
    ).drop(columns=["idhogar"])

    # Fill in the individuals that do not have a head of household with 4 since these will not be scored
    submission["Target"] = submission["Target"].fillna(4).astype(np.int8)

    # return the submission and feature importances along with validation scores
    return submission, feature_importances, valid_scores


def cv_model(train, train_labels, model, scorer, name, model_results=None):
    """Perform 10 fold cross validation of a model"""

    cv_scores = cross_val_score(model, train, train_labels, cv=10, scoring=scorer, n_jobs=-1)
    print(f"10 Fold CV Score: {round(cv_scores.mean(), 5)} with std: {round(cv_scores.std(), 5)}")

    if model_results is not None:
        model_results = pd.concat(
            [
                model_results,
                pd.DataFrame(
                    {"model": name, "cv_mean": cv_scores.mean(), "cv_std": cv_scores.std()},
                    index=[0],
                ),
            ],
            axis=0,
            ignore_index=True,
        )

        return model_results


### scorer ###
scorer_params = {"score_func": f1_score, "greater_is_better": True, "average": "macro"}

### RF ###

rf_params = {"n_estimators": 100, "random_state": 10, "n_jobs": -1}
