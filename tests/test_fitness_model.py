import hashlib
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data_processing import ASSAY_ID, write_csv, write_json
from evaluate_fitness import regression_metrics, run as evaluate, tie_priority, top_k_metrics
from features import LABELED_FIELDS, SPLIT_SIZES
from fitness_model import FitnessPredictor
from train_fitness import run, select_trial


class FitnessModelTests(unittest.TestCase):
    def test_count_baseline_training_statistics_and_fallback(self):
        reference = "A" * 20
        model = FitnessPredictor("count_only", reference).fit([
            {"sequence": "C"*6+"A"*14, "fitness": 1, "fitness_bin": 0},
            {"sequence": "D"*6+"A"*14, "fitness": 3, "fitness_bin": 1},
            {"sequence": "CA"+"A"*18, "fitness": 5, "fitness_bin": 1},
        ])
        p, levels = model.predict_with_fallback([6, 7, 8, 16, 1])
        np.testing.assert_array_equal(p, [2, 2, 3, 3, 5])
        self.assertEqual(levels, ["exact_count", "stratification_group", "global_mean",
                                  "global_mean", "exact_count"])
        self.assertEqual(model.predict("C"*7+"A"*13), 2)

    def test_top_k_uses_label_free_tie_break(self):
        priority = tie_priority(["a", "b", "c", "d"])
        best_priority = int(np.argmin(priority))
        y = np.ones(4)
        y[best_priority] = 0
        result = top_k_metrics(y, np.zeros(4), priority, ks=(1, 2))
        self.assertEqual(result[0]["hit_rate"], 0)
        self.assertEqual(result[0]["mean_experimental_fitness"], 0)
        self.assertEqual(result[1]["hit_rate"], 0.5)
        self.assertEqual(result[1]["max_experimental_fitness"], 1)
        order = np.array([3, 2, 1, 0])
        self.assertEqual(top_k_metrics(y[order], np.zeros(4), priority[order], ks=(1, 2)), result)

    def test_metrics_on_constant_prediction(self):
        result = regression_metrics(np.array([1., 2.]), np.array([2., 2.]), overall=True)
        self.assertIsNone(result["spearman"])
        self.assertIsNone(result["pearson"])
        self.assertEqual(result["undefined_correlation_reason"], "constant predicted fitness")
        self.assertEqual(result["mse"], 0.5)
        self.assertEqual(result["r2"], -1)

    def test_validation_selection_near_tie(self):
        def trial(name, key, overall, high, params):
            return {"model_name": name, "trial_id": key, "params": params,
                    "validation": {"overall": {"spearman": overall}, "high_fitness": {"spearman": high}}}

        ridge = trial("ridge", "r", 0.795, 0.62, {"alpha": 1})
        forest = trial("random_forest", "f", 0.80, 0.58,
                       {"n_estimators": 100, "max_depth": 15, "min_samples_leaf": 1, "max_features": 1.0})
        count = trial("count_only", "c", 0.785, 0.9, {})
        self.assertIs(select_trial([count, ridge, forest])[0], ridge)
        ridge["validation"]["high_fitness"]["spearman"] = 0.58
        self.assertIs(select_trial([count, ridge, forest])[0], ridge)

    def test_prediction_reproducibility_fixed_seed_and_model_reload(self):
        training = [{"sequence": seq, "fitness": score}
                    for seq, score in zip(["ACD", "ADD", "WCD", "ACC", "ACG"], [2, 1, 0, 4, 3])]
        probe = ["ACD", "ACF", "WCD"]
        parameters = {
            "count_only": {},
            "ridge": {"solver": "lsqr", "alpha": 1},
            "random_forest": {"random_state": 42, "n_estimators": 5, "n_jobs": 1},
        }
        with TemporaryDirectory(prefix=".gfp-model-test-", dir=ROOT) as temporary:
            for name, params in parameters.items():
                with self.subTest(name=name):
                    model = FitnessPredictor(name, "ACD", params).fit(training)
                    expected = model.predict(probe, batch_size=2)
                    path = Path(temporary) / f"{name}.joblib"
                    model.save(path)
                    loaded = FitnessPredictor.load(path)
                    np.testing.assert_allclose(loaded.predict(iter(probe), batch_size=1), expected, atol=1e-12)
                    np.testing.assert_allclose(loaded.fit(training).predict(probe), expected, atol=1e-12)
                    self.assertAlmostEqual(loaded.predict(probe[0]), expected[0])

    def test_training_io_never_opens_test_or_oracle(self):
        with TemporaryDirectory(prefix=".gfp-model-test-", dir=ROOT) as temporary:
            root = Path(temporary)
            processed = root / "data" / "processed"
            processed.mkdir(parents=True)
            splits = root / "data" / "splits"
            splits.mkdir()
            (processed / "gfp_wildtype.fasta").write_text(f">{ASSAY_ID}|reference\n" + "A"*238 + "\n")
            write_json(processed / "gfp_metadata.json", {"dataset_name": ASSAY_ID, "sequence_length": 238,
                                                        "binarization_cutoff_source_value": "2.5"})

            def make_row(index):
                pos = index + 1
                count = 1 + index % 2
                sequence = "A"*index + "C"*count + "A"*(238-index-count)
                score = 4 - count + (index % 7) / 10
                return {"experiment_variant_id": hashlib.sha256(sequence.encode("ascii")).hexdigest(),
                        "mutation_count": count, "mutant": ":".join(f"A{pos+i}C" for i in range(count)),
                        "sequence": sequence, "mutation_positions": ";".join(str(pos+i) for i in range(count)),
                        "wt_amino_acids": ";".join("A"*count), "mutant_amino_acids": ";".join("C"*count),
                        "fitness": score, "fitness_bin": int(score >= 2.5)}

            write_csv(splits / "train_initial.csv", LABELED_FIELDS, map(make_row, range(40)))
            write_csv(splits / "validation.csv", LABELED_FIELDS, map(make_row, range(40, 140)))
            forbidden = {"test.csv", "oracle_labels.csv", "candidate_pool.csv", "eda_summary.json"}
            real_open = Path.open

            def guarded_open(path, *args, **kwargs):
                self.assertNotIn(path.name, forbidden)
                return real_open(path, *args, **kwargs)

            grid = {"count_only": ({},), "ridge": ({"solver": "lsqr", "alpha": 1},),
                    "random_forest": ({"random_state": 42, "n_estimators": 5, "n_jobs": 1},)}
            with patch.dict(SPLIT_SIZES, {"train_initial": 40, "validation": 100}), \
                    patch("train_fitness.grids", return_value=grid), patch.object(Path, "open", guarded_open):
                run(root)
            selection = json.loads((root / "results" / "fitness_model" / "model_selection.json").read_text())
            self.assertEqual(selection["status"], "locked")
            self.assertTrue(forbidden.isdisjoint(Path(path).name for path in selection["locked_files_sha256"]))

    def test_evaluation_rejects_missing_selection_before_loading_splits(self):
        with TemporaryDirectory(prefix=".gfp-model-test-", dir=ROOT) as temporary:
            with patch("evaluate_fitness.load_split") as load_split:
                with self.assertRaises(FileNotFoundError):
                    evaluate(Path(temporary), candidate_smoke=True)
                load_split.assert_not_called()


if __name__ == "__main__":
    unittest.main()
