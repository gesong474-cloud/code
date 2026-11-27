"""Reusable training, batch prediction and persistence for GFP fitness models."""

from collections import defaultdict
from itertools import islice
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

from features import SequenceEncoder

MODEL_NAMES = ("count_only", "ridge", "random_forest")


def count_group(count):
    if 1 <= count <= 5:
        return str(count)
    if 6 <= count <= 7:
        return "6-7"
    if 8 <= count <= 15:
        return "8-15"
    return None


class FitnessPredictor:
    def __init__(self, model_name, reference, params=None):
        if model_name not in MODEL_NAMES:
            raise ValueError(f"Unknown model family {model_name!r}")
        self.model_name = model_name
        self.encoder = SequenceEncoder(reference)
        self.params = dict(params or {})
        self.estimator = None
        self.count_means = None

    def fit(self, training_data):
        """Fit from explicitly supplied dictionaries containing sequence and experimental fitness."""
        rows = list(training_data)
        return self.fit_features(self.encode([row["sequence"] for row in rows]),
                                 [row["fitness"] for row in rows])

    def encode(self, sequences):
        if self.model_name == "count_only":
            return self.encoder.mutation_counts(sequences)
        return self.encoder.transform(sequences, include_count=self.model_name == "random_forest")

    def fit_features(self, features, fitness):
        target = np.asarray(fitness, dtype=np.float64)
        if target.ndim != 1 or len(target) == 0 or not np.isfinite(target).all():
            raise ValueError("Expected nonempty finite experimental fitness values")
        if features.shape[0] != len(target):
            raise ValueError("Training features and labels have different numbers of rows")
        if self.model_name == "count_only":
            by_count, by_group = defaultdict(list), defaultdict(list)
            for count, score in zip(features, target):
                by_count[int(count)].append(score)
                group = count_group(count)
                if group is not None:
                    by_group[group].append(score)
            self.count_means = {key: float(np.mean(scores)) for key, scores in by_count.items()}
            self.group_means = {key: float(np.mean(scores)) for key, scores in by_group.items()}
            self.global_mean = float(target.mean())
        else:
            estimator_type = Ridge if self.model_name == "ridge" else RandomForestRegressor
            self.estimator = estimator_type(**self.params).fit(features, target)
        return self

    def predict_features(self, features):
        if self.model_name != "count_only":
            if self.estimator is None:
                raise ValueError("Fit the model before prediction")
            return self.estimator.predict(features).astype(np.float64, copy=False)
        return self.predict_with_fallback(features)[0]

    def predict_with_fallback(self, counts):
        if self.count_means is None:
            raise ValueError("Fit the Count-only model before prediction")
        predictions, levels = [], []
        for count in counts:
            group = count_group(count)
            if count in self.count_means:
                score, level = self.count_means[count], "exact_count"
            elif group in self.group_means:
                score, level = self.group_means[group], "stratification_group"
            else:
                score, level = self.global_mean, "global_mean"
            predictions.append(score)
            levels.append(level)
        return np.asarray(predictions), levels

    def predict(self, sequences, batch_size=1024):
        """Return a float for a single sequence or a float64 array for an iterable."""
        if batch_size < 1:
            raise ValueError("Prediction batch size must be positive")
        single = isinstance(sequences, str)
        source = iter([sequences] if single else sequences)
        predictions = []
        while batch := list(islice(source, batch_size)):
            predictions.append(self.predict_features(self.encode(batch)))
        joined = np.concatenate(predictions) if predictions else np.empty(0, dtype=np.float64)
        return float(joined[0]) if single else joined

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path, compress=3)

    @classmethod
    def load(cls, path):
        """Load a trusted locally generated joblib artifact."""
        model = joblib.load(path)
        if not isinstance(model, cls):
            raise ValueError(f"Expected a FitnessPredictor artifact at {path}")
        return model
