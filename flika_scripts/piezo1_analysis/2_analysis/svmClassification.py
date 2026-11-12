"""Qt-free SVM track classification used by Step 2 (and the web interface).

predict_SPT_class trains an RBF support-vector classifier (optional Box-Cox feature transform, PCA to 3 components,
standard scaling, grid search over C and gamma with 10-fold cross-validation) on a hand-labelled training-features
file and writes a *_tracksRG.csv file back with an SVM column: 1 mobile, 2 intermediate (confined), 3 trapped.

feature_transform (default pipeline_defaults.FEATURE_TRANSFORM):
  "boxcox"  ShiftedBoxCox is the first pipeline step: its shifts and lambdas are fitted on the training features (in
            each cross-validation fold, then on the whole training split) and reused unchanged at prediction.
  "none"    raw features (legacy: the old code's Box-Cox never took effect, so published results used raw features).
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn import decomposition
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.svm import SVC

from pipeline_defaults import FEATURE_TRANSFORM, FEATURE_TRANSFORMS

FEATURES = ['NetDispl', 'Straight', 'Asymmetry', 'radiusGyration', 'Kurtosis', 'fracDimension']
LABELS = {"mobile": 1, "confined": 2, "trapped": 3}
TRAIN_EXPERIMENT = 'tdTomato_37Degree'
PARAM_GRID = {
    "pca__n_components": [3],
    "SVC__C": [0.1, 0.5, 1, 10, 30, 40, 50, 75, 100, 500, 1000],
    "SVC__gamma": [0.001, 0.005, 0.01, 0.03, 0.05, 0.1, 0.5, 1., 5., 10., 50.0],
}
PRED_RENAME = {"radius_gyration": "radiusGyration", "track_number": "TrackID", "netDispl": "NetDispl",
               "asymmetry": "Asymmetry", "kurtosis": "Kurtosis"}


class ShiftedBoxCox(BaseEstimator, TransformerMixin):
    """Box-Cox with standardisation, fitted on training features only.

    As the original prepare_box_cox_data: a column whose training minimum is <= 0 is shifted by |min| + 1e-15 so it
    is strictly positive. Shifts, lambdas and the standardisation are stored at fit and reused by transform. A
    prediction value below the training minimum (which the shift cannot make positive) is clipped to that minimum.
    Fits are memoised by the training data's bytes: a grid search refits the same 10 folds for every (C, gamma).
    """
    _cache = {}

    def fit(self, X, y=None):
        X = np.ascontiguousarray(X, dtype=float)
        key = (X.shape, X.tobytes())
        if key not in self._cache:
            lo = X.min(axis=0)
            shift = np.where(lo <= 0, np.abs(lo) + 1e-15, 0.0)
            if len(self._cache) > 64:
                self._cache.clear()
            self._cache[key] = (shift, lo + shift, PowerTransformer(method='box-cox').fit(X + shift))
        self.shift_, self.floor_, self.power_ = self._cache[key]
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return self.power_.transform(np.maximum(X + self.shift_, self.floor_))


def check_transform(feature_transform):
    ft = FEATURE_TRANSFORM if feature_transform is None else str(feature_transform).lower()
    if ft not in FEATURE_TRANSFORMS:
        raise ValueError(f"feature_transform must be one of {FEATURE_TRANSFORMS}, not {feature_transform!r}")
    return ft


def make_pipeline(feature_transform=None):
    steps = [("pca", decomposition.PCA()), ("scaler", StandardScaler()), ("SVC", SVC(kernel="rbf"))]
    if check_transform(feature_transform) == "boxcox":
        steps.insert(0, ("boxcox", ShiftedBoxCox()))
    return Pipeline(steps)


def load_training(train_data_path):
    """Training features (FEATURES columns) and labels (1/2/3) of the tdTomato_37Degree rows."""
    train_feats = pd.read_csv(Path(train_data_path), sep=',')
    train_feats = train_feats.loc[train_feats['Experiment'] == TRAIN_EXPERIMENT]
    X = train_feats[FEATURES]
    y = train_feats['Elected_Label'].replace(LABELS).astype(int)  # replace() keeps object dtype in pandas >= 3
    return X, y


def train_classifier(X, y, feature_transform=None, cv=10):
    """Grid-search the pipeline on an 80 % training split (random_state 0, as before) and refit with the best
    parameters. Returns (pipeline, search, (X_test, y_test)); search.best_score_ is the cross-validated accuracy."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
    search = GridSearchCV(make_pipeline(feature_transform), param_grid=PARAM_GRID, cv=cv)
    search.fit(X_train, y_train)
    pipeline = make_pipeline(feature_transform).set_params(**search.best_params_)
    pipeline.fit(X_train, y_train)
    return pipeline, search, (X_test, y_test)


def predict_SPT_class(train_data_path, pred_data_path, exptName, level, feature_transform=None):
    """Computes predicted class for SPT data where
        1:Mobile, 2:Intermediate, 3:Trapped

    Args:
        train_data_path (str): complete path to training features data file in .csv format, ex. 'C:/data/tdTomato_37Degree_CytoD_training_feats.csv'
                               should be a .csv file with features columns, an 'Experiment' column identifying the unique experiment ('tdTomato_37Degree'),
                               a 'TrackID' column with unique numerical IDs for each track, and an 'Elected_Label' column derived from human voting.
        pred_data_path (str): complete path to the *_tracksRG.csv features file that needs predictions.
        exptName (str): written to the 'Experiment' column.
        level (str): suffix of the output name.
        feature_transform (str): "boxcox" or "none" (legacy); default pipeline_defaults.FEATURE_TRANSFORM.

    Output:
        <pred_data_path stem>_SVMPredicted<level>.csv next to the input: the input table plus 'Experiment' and 'SVM'.
    """
    pipeline, _, _ = train_classifier(*load_training(train_data_path), feature_transform=feature_transform)

    tracksDF = pd.read_csv(Path(pred_data_path), sep=',')
    X = tracksDF.rename(columns=PRED_RENAME)
    tracksDF['Experiment'] = exptName
    tracksDF['SVM'] = pipeline.predict(X[FEATURES]).astype('int')
    X_outpath = Path(pred_data_path).parents[0] / f'{Path(pred_data_path).stem}_SVMPredicted{level}.csv'
    tracksDF.to_csv(X_outpath, sep=',', index=None)
