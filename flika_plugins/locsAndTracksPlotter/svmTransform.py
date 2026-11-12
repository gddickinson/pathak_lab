"""Box-Cox feature transform for the SVM track classifier (used by joinTracks.JoinTracks.predict_SPT_class).

Copy of ShiftedBoxCox in flika_scripts/piezo1_analysis/2_analysis/svmClassification.py (the plugin is installed on
its own, so it cannot import the pipeline scripts); tests/test_svm_transform.py checks the two copies agree.
FEATURE_TRANSFORM mirrors 2_analysis/pipeline_defaults.py: 'boxcox' (default) or 'none' (legacy, as published).
"""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PowerTransformer

FEATURE_TRANSFORMS = ("boxcox", "none")
FEATURE_TRANSFORM = "boxcox"


class ShiftedBoxCox(BaseEstimator, TransformerMixin):
    """Box-Cox with standardisation, fitted on training features only.

    A column whose training minimum is <= 0 is shifted by |min| + 1e-15 so it is strictly positive. Shifts, lambdas
    and the standardisation are stored at fit and reused by transform. A prediction value below the training minimum
    (which the shift cannot make positive) is clipped to that minimum.
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
