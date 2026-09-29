"""
Models. The learners come from the two modelling notebooks
(LogisticRegression, RandomForest, XGBoost, LightGBM); DecisionTree is left out
because a single tree was the weakest and least stable of them.

Differences from the notebooks
------------------------------
* Corner-swap augmentation + symmetric prediction:
      p(A beats B) = ( f(x) + 1 - f(-x) ) / 2
  so the answer never depends on which corner a fighter is listed in.
* Soft-voting ensemble whose weights are learned from graded predictions
  (see learning.py), followed by temperature scaling for calibration.
* A second model predicts the method (KO/TKO, Submission, Decision) given
  who wins, which yields the six "most probable outcomes".
* XGBoost / LightGBM are optional: if they aren't installed, sklearn's
  HistGradientBoosting covers the boosted-tree slot.
"""
from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import METHODS

def _importable(name: str) -> bool:
    """True only if the package actually imports. On macOS xgboost/lightgbm install fine but fail to
    load without Homebrew's libomp, so checking find_spec alone is not enough."""
    try:
        importlib.import_module(name)
        return True
    except Exception:
        return False


HAS_XGB = _importable("xgboost")
HAS_LGBM = _importable("lightgbm")
HAS_CAT = _importable("catboost")

LEARNER_NAMES = {"logreg": "Logistic regression", "rf": "Random forest",
                 "hgb": "Hist. gradient boosting", "xgb": "XGBoost", "lgbm": "LightGBM", "cat": "CatBoost"}


def available_learners() -> list[str]:
    out = ["logreg", "rf", "hgb"]
    if HAS_XGB:
        out.append("xgb")
    if HAS_LGBM:
        out.append("lgbm")
    if HAS_CAT:
        out.append("cat")
    return out


def preset(name: str) -> list[str]:
    avail = available_learners()
    if name == "fast":
        return ["logreg", "lgbm" if HAS_LGBM else "hgb"]
    if name == "balanced":
        # Random forest is left out on purpose: in walk-forward tests it was the weakest member and
        # slightly hurt the blend. Boosted learners join automatically when installed.
        return [l for l in ["logreg", "hgb", "xgb", "lgbm", "cat"] if l in avail]
    return avail  # "full" (everything, including random forest)


def make_learner(name: str, seed: int = 42):
    if name == "logreg":
        return Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
                         ("clf", LogisticRegression(C=0.05, max_iter=3000))])
    if name == "rf":
        return Pipeline([("imp", SimpleImputer(strategy="median")),
                         ("clf", RandomForestClassifier(n_estimators=250, min_samples_leaf=25,
                                                        max_features="sqrt", n_jobs=-1, random_state=seed))])
    if name == "hgb":
        return HistGradientBoostingClassifier(max_iter=250, learning_rate=0.04, max_leaf_nodes=15,
                                              min_samples_leaf=40, l2_regularization=1.0,
                                              early_stopping=False, random_state=seed)
    if name == "xgb":
        import xgboost as xgb
        return xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.04, subsample=0.8,
                                 colsample_bytree=0.7, min_child_weight=10, reg_lambda=2.0,
                                 eval_metric="logloss", n_jobs=-1, random_state=seed)
    if name == "lgbm":
        import lightgbm as lgb
        return lgb.LGBMClassifier(n_estimators=300, learning_rate=0.04, num_leaves=15, min_child_samples=40,
                                  subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=2.0,
                                  verbose=-1, random_state=seed)
    if name == "cat":
        from catboost import CatBoostClassifier
        # Ordered boosting + symmetric trees: robust on small, noisy tabular data like fight records.
        return CatBoostClassifier(iterations=600, depth=5, learning_rate=0.04, l2_leaf_reg=5.0,
                                  subsample=0.8, bootstrap_type="Bernoulli", loss_function="Logloss",
                                  allow_writing_files=False, verbose=0, thread_count=-1, random_seed=seed)
    raise ValueError(f"unknown learner {name}")


def _fit(est, X, y, w):
    if w is None:
        return est.fit(X, y)
    if isinstance(est, Pipeline):
        return est.fit(X, y, **{f"{est.steps[-1][0]}__sample_weight": w})
    return est.fit(X, y, sample_weight=w)


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def blend(components: pd.DataFrame, weights: dict | None = None, temperature: float = 1.0) -> np.ndarray:
    names = list(components.columns)
    w = np.array([(weights or {}).get(n, 1.0) for n in names], dtype=float)
    w = w / w.sum() if w.sum() > 0 else np.full(len(names), 1 / len(names))
    p = components.to_numpy(float) @ w
    if temperature != 1.0:
        p = _sigmoid(temperature * _logit(p))
    return np.clip(p, 1e-4, 1 - 1e-4)


class WinModel:
    def __init__(self, learners: list[str] | None = None, seed: int = 42):
        self.learners = learners or preset("balanced")
        self.seed = seed
        self.models: dict = {}
        self.diff_cols: list[str] = []
        self.ctx_cols: list[str] = []

    def _X(self, D: pd.DataFrame, C: pd.DataFrame, flip: bool = False) -> np.ndarray:
        d = D.reindex(columns=self.diff_cols).to_numpy(float)
        return np.hstack([-d if flip else d, C.reindex(columns=self.ctx_cols).to_numpy(float)])

    def fit(self, D: pd.DataFrame, C: pd.DataFrame, y, sample_weight=None) -> "WinModel":
        self.diff_cols, self.ctx_cols = list(D.columns), list(C.columns)
        y = np.asarray(y, dtype=int)
        X = np.vstack([self._X(D, C), self._X(D, C, flip=True)])
        yy = np.r_[y, 1 - y]
        ww = None if sample_weight is None else np.r_[sample_weight, sample_weight].astype(float)
        for name in self.learners:
            self.models[name] = _fit(make_learner(name, self.seed), X, yy, ww)
        return self

    def predict_components(self, D: pd.DataFrame, C: pd.DataFrame) -> pd.DataFrame:
        Xa, Xb = self._X(D, C), self._X(D, C, flip=True)
        out = {n: 0.5 * (m.predict_proba(Xa)[:, 1] + 1 - m.predict_proba(Xb)[:, 1])
               for n, m in self.models.items()}
        return pd.DataFrame(out, index=D.index)

    def explain(self, D: pd.DataFrame, C: pd.DataFrame, top: int = 6) -> list[list[tuple[str, float]]]:
        """Per-row top feature contributions (log-odds, + favours A) from the logistic model."""
        m = self.models.get("logreg")
        if m is None:
            return [[] for _ in range(len(D))]
        Xa = self._X(D, C)
        z = m.named_steps["sc"].transform(m.named_steps["imp"].transform(Xa))
        contrib = z * m.named_steps["clf"].coef_[0]
        names = self.diff_cols + self.ctx_cols
        n_diff = len(self.diff_cols)
        res = []
        for row in contrib:
            order = np.argsort(-np.abs(row[:n_diff]))[:top]
            res.append([(names[j], float(row[j])) for j in order if abs(row[j]) > 1e-3])
        return res


class MethodModel:
    """P(method | winner) with winner-oriented features."""

    def __init__(self, seed: int = 42):
        self.model = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
                                                    min_samples_leaf=40, l2_regularization=1.0,
                                                    early_stopping=False, random_state=seed)
        self.cols: list[str] = []
        self.prior = np.array([1 / 3] * 3)

    def fit(self, X: pd.DataFrame, method: pd.Series, sample_weight=None) -> "MethodModel":
        self.cols = list(X.columns)
        m = method.astype(str).to_numpy()
        self.prior = np.array([(m == c).mean() for c in METHODS])
        self.model.fit(X.to_numpy(float), m, sample_weight=sample_weight)
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        P = self.model.predict_proba(X.reindex(columns=self.cols).to_numpy(float))
        out = pd.DataFrame(0.0, index=X.index, columns=METHODS)
        for j, c in enumerate(self.model.classes_):
            if c in out.columns:
                out[c] = P[:, j]
        return out.div(out.sum(axis=1).replace(0, 1), axis=0)


def fit_temperature(p, y) -> float:
    """Scalar a minimising log loss of sigmoid(a * logit(p)); a<1 = was over-confident."""
    from scipy.optimize import minimize_scalar
    p, y = np.asarray(p, float), np.asarray(y, float)
    if len(p) < 150:
        return 1.0
    z = _logit(p)

    def nll(a):
        q = np.clip(_sigmoid(a * z), 1e-6, 1 - 1e-6)
        return -np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))

    return float(minimize_scalar(nll, bounds=(0.3, 2.5), method="bounded").x)


def hedge_weights(components: pd.DataFrame, y, eta: float = 0.05, floor: float = 0.03) -> dict:
    """Exponential-weights (Hedge) update from each learner's cumulative log loss."""
    y = np.asarray(y, float)
    P = np.clip(components.to_numpy(float), 1e-6, 1 - 1e-6)
    ll = -(y[:, None] * np.log(P) + (1 - y[:, None]) * np.log(1 - P)).sum(axis=0)
    w = np.exp(-eta * (ll - ll.min()))
    w = w / w.sum()
    w = np.maximum(w, floor)
    w = w / w.sum()
    return dict(zip(components.columns, map(float, w)))
