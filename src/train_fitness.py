"""Fit Stage 3 models using training labels and lock a validation-only decision."""

import argparse
import platform
from collections import Counter
from pathlib import Path
from shutil import copyfile
from time import perf_counter

import joblib
import matplotlib
import numpy as np
import scipy
import sklearn

from data_processing import AMINO_ACIDS, file_sha256, write_csv, write_json
from evaluate_fitness import PRED_FIELDS, SEED, describe_predictions, prediction_records
from features import REFERENCE_INPUTS, check_disjoint, load_reference, load_split
from fitness_model import MODEL_NAMES, FitnessPredictor

RIDGE_ALPHAS = (0.01, 0.1, 1, 10, 100)
RF_GRID = (
    (100, 15, 1, 0.3), (100, 15, 3, 1.0), (100, 30, 1, 1.0), (100, None, 3, 0.3),
    (200, 15, 3, 0.3), (200, 30, 3, 1.0), (200, 30, 1, 0.3), (200, None, 1, 1.0),
)
TOLERANCE = 0.01
TRAIN_PATHS = ("data/splits/train_initial.csv", "data/splits/validation.csv", *REFERENCE_INPUTS)
RULE = ("Retain configurations within 0.01 of the best defined validation Spearman within the comparison, "
        "then maximize defined high-fitness validation Spearman. For remaining ties favor Count-only, "
        "Ridge, then Random Forest. Within a family prefer higher Ridge alpha or fewer and shallower "
        "trees with larger leaves and fewer candidate features. Undefined correlations rank last. "
        "If every overall Spearman is undefined, compare all configs on high-fitness Spearman and simplicity.")


def grids():
    return {
        "count_only": ({},),
        "ridge": tuple({"alpha": alpha, "solver": "lsqr", "tol": 1e-7, "max_iter": 3000}
                       for alpha in RIDGE_ALPHAS),
        "random_forest": tuple(
            {"n_estimators": n, "max_depth": depth, "min_samples_leaf": leaf, "max_features": features,
             "random_state": SEED, "n_jobs": 2}
            for n, depth, leaf, features in RF_GRID
        ),
    }


def simplicity(trial):
    model, params = trial["model_name"], trial["params"]
    if model == "ridge":
        return (1, -params.get("alpha", 1))
    if model == "random_forest":
        return (2, params.get("n_estimators", 100), params.get("max_depth") or float("inf"),
                -params.get("min_samples_leaf", 1), params.get("max_features", 1.0))
    return (0,)


def select_trial(trials):
    scorers = [trial["validation"]["overall"]["spearman"] for trial in trials]
    defined = [score for score in scorers if score is not None]
    best = max(defined) if defined else None
    eligible = [trial for score, trial in zip(scorers, trials)
                if best is None or (score is not None and best - score <= TOLERANCE + 1e-12)]

    def key(trial):
        high = trial["validation"]["high_fitness"]["spearman"]
        return (-(high if high is not None else -float("inf")), simplicity(trial), trial["trial_id"])

    chosen = min(eligible, key=key)
    return chosen, {"best_overall_spearman": best,
                    "eligible_trial_ids": [trial["trial_id"] for trial in eligible],
                    "chosen_trial_id": chosen["trial_id"]}


def feature_size(matrix):
    return {"shape": list(matrix.shape), "dtype": str(matrix.dtype),
            "format": matrix.format, "stored_values": matrix.nnz,
            "matrix_bytes": matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes}


def run(root):
    start_total = perf_counter()
    output = root / "results" / "fitness_model"
    output.mkdir(parents=True, exist_ok=True)
    lock = output / "model_selection.json"
    # Invalidate an older lock before replacing any training artifacts.
    lock.unlink(missing_ok=True)
    encoder, cutoff = load_reference(root)
    training = load_split(root, "train_initial", encoder)
    validation = load_split(root, "validation", encoder)
    check_disjoint({"train_initial": training, "validation": validation})
    train_seq = [row["sequence"] for row in training]
    val_seq = [row["sequence"] for row in validation]
    y = np.array([row["fitness"] for row in training], dtype=np.float64)
    training_candidates, families, decisions, records, logs = [], {}, {}, [], []
    features = {}
    for name, configurations in grids().items():
        started = perf_counter()
        extractor = FitnessPredictor(name, encoder.reference)
        x, v = extractor.encode(train_seq), extractor.encode(val_seq)
        seconds = perf_counter() - started
        if name != "count_only":
            features["one_hot" if name == "ridge" else "one_hot_and_mutation_count"] = {
                "train": feature_size(x), "validation": feature_size(v),
                "estimated_max_candidate_csr_bytes_per_1024_rows": 1024 * (x.shape[1] // 20
                    + int(name == "random_forest")) * 8 + 1025 * 4,
                "encoding_seconds_train_and_validation": seconds,
            }
        trials, fitted, predictions, fallback = [], {}, {}, {}
        for i, params in enumerate(configurations, start=1):
            trial_id = f"{name}_{i:02d}"
            model = FitnessPredictor(name, encoder.reference, params)
            start = perf_counter()
            model.fit_features(x, y)
            elapsed_fit = perf_counter() - start
            start = perf_counter()
            estimates = model.predict_features(v)
            elapsed_pred = perf_counter() - start
            fitted[trial_id], predictions[trial_id] = model, estimates
            if name == "count_only":
                _, levels = model.predict_with_fallback(v)
                fallback = dict(Counter(levels))
            trial = {
                "trial_id": trial_id, "model_name": name, "params": params,
                "training_seconds": elapsed_fit, "validation_prediction_seconds": elapsed_pred,
                "validation": describe_predictions(validation, estimates, cutoff),
            }
            trials.append(trial)
            scores = trial["validation"]
            logs.append({"trial_id": trial_id, "model_name": name, "parameters": repr(params),
                         "training_seconds": elapsed_fit, "validation_prediction_seconds": elapsed_pred,
                         **{k: scores["overall"][k] for k in ("spearman", "pearson", "mse", "rmse", "mae", "r2")},
                         "high_fitness_spearman": scores["high_fitness"]["spearman"],
                         **{f"top_{item['k']}_hit_rate": item["hit_rate"] for item in scores["top_k"]}})
            score = scores["overall"]["spearman"]
            score_text = "NA" if score is None else f"{score:.5f}"
            print(f"{trial_id}: validation Spearman={score_text}, "
                  f"fit={elapsed_fit:.2f}s", flush=True)
        best, decisions[name] = select_trial(trials)
        trial_id = best["trial_id"]
        model_path = f"models/{name}.joblib"
        fitted[trial_id].save(root / model_path)
        families[name] = {**best, "model_path": model_path}
        if fallback:
            families[name]["validation_fallback_counts"] = fallback
        records.extend(prediction_records(validation, predictions[trial_id], name, "validation"))
        training_candidates.extend(trials)

    selected_trial, decisions["across_families"] = select_trial(list(families.values()))
    selected = selected_trial["model_name"]
    final_path = "models/fitness_predictor.joblib"
    copyfile(root / families[selected]["model_path"], root / final_path)
    probe = val_seq[:32]
    before = np.array([row["predicted_fitness"] for row in records if row["model_name"] == selected])[:32]
    after = FitnessPredictor.load(root / final_path).predict(probe)
    reload_delta = float(np.max(np.abs(before - after)))
    if not np.allclose(before, after, rtol=0, atol=1e-12):
        raise ValueError("Selected model prediction changed after serialization")
    write_csv(output / "validation_predictions.csv", PRED_FIELDS, records)
    write_csv(output / "hyperparameter_search.csv", tuple(logs[0]), logs)
    software = {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                "scikit_learn": sklearn.__version__, "joblib": joblib.__version__,
                "matplotlib": matplotlib.__version__}
    summary = {
        "training_samples": len(training), "validation_samples": len(validation), "seed": SEED,
        "software_versions": software, "feature_memory": features, "amino_acid_order": AMINO_ACIDS,
        "sequence_length": encoder.length, "reference_input": REFERENCE_INPUTS[0],
        "reference_sequence": encoder.reference,
        "reference_file_sha256": file_sha256(root / REFERENCE_INPUTS[0]),
        "features_used": {"count_only": ["mutation_count"], "ridge": ["position_major_one_hot"],
                          "random_forest": ["position_major_one_hot", "mutation_count"]},
        "selected_reload_consistency": {"status": "passed", "max_absolute_difference": reload_delta,
                                        "absolute_tolerance": 1e-12},
        "training_pipeline_seconds": perf_counter() - start_total,
        "serialized_model_bytes": {name: (root / path).stat().st_size for name, path in
                                   {**{n: v["model_path"] for n, v in families.items()}, "selected": final_path}.items()},
    }
    write_json(output / "training_summary.json", summary)
    paths = (*TRAIN_PATHS, *(families[name]["model_path"] for name in MODEL_NAMES), final_path,
             "results/fitness_model/validation_predictions.csv")
    write_json(lock, {
        "status": "locked", "seed": SEED, "threshold": cutoff, "selection_rule": RULE,
        "top_k_tie_breaking": "SHA-256 of seed:experiment_variant_id; independent of observed fitness",
        "selected_model": selected, "selected_model_path": final_path,
        "selected_params": selected_trial["params"], "decisions": decisions, "families": families,
        "trials": training_candidates, "software_versions": software,
        "locked_files_sha256": {path: file_sha256(root / path) for path in paths},
        "test_and_candidate_input_hashes": "Recorded in evaluation_summary.json after this selection is saved.",
    })
    print(f"Locked {selected} ({selected_trial['trial_id']}); training and validation finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    run(parser.parse_args().root.resolve())
