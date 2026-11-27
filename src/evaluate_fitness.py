"""Evaluate fitness predictions and run the locked independent-test stage."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter
from scipy.stats import rankdata

from data_processing import file_sha256, read_csv, write_csv, write_json
from features import ID, check_disjoint, load_reference, load_split
from fitness_model import MODEL_NAMES, FitnessPredictor

SEED = 42
KS = (20, 50, 100)
GROUPS = ("1", "2", "3", "4-5", "6-7", "8-15")
DISPLAY = ("Count-only", "Ridge", "Random Forest")
COLORS = ("#266794", "#a44931", "#4b7954")
PRED_FIELDS = (ID, "mutant", "mutation_count", "experimental_fitness", "predicted_fitness",
               "residual", "absolute_error", "model_name", "split")


def correlation(a, b):
    if len(a) < 2:
        return None, "fewer than two observations"
    if np.all(a == a[0]):
        return None, "constant experimental fitness"
    if np.all(b == b[0]):
        return None, "constant predicted fitness"
    return float(np.clip(np.corrcoef(a, b)[0, 1], -1, 1)), None


def regression_metrics(y, p, overall=False):
    spearman, reason = correlation(rankdata(y), rankdata(p))
    pearson, _ = correlation(y, p)
    errors = p - y
    mse = float(np.mean(np.square(errors))) if len(errors) else None
    result = {"sample_count": len(y), "spearman": spearman, "pearson": pearson,
              "rmse": np.sqrt(mse).item() if mse is not None else None,
              "mae": float(np.mean(np.abs(errors))) if len(errors) else None,
              "undefined_correlation_reason": reason}
    if overall:
        variance = float(np.var(y)) if len(y) >= 2 else 0
        result.update(mse=mse, r2=1 - mse / variance if variance else None)
    return result


def tie_priority(identifiers, seed=SEED):
    # The order is independent of row order, fitness and the estimator.
    return np.array([hashlib.sha256(f"{seed}:{identifier}".encode("ascii")).hexdigest()
                     for identifier in identifiers])


def top_k_metrics(y, p, priority, ks=KS):
    predicted = np.lexsort((priority, -p))
    actual = np.lexsort((priority, -y))
    results = []
    for k in ks:
        if not 1 <= k <= len(y):
            raise ValueError(f"k={k} is invalid for an evaluation set of {len(y)} rows")
        top = predicted[:k]
        observed = y[top]
        results.append({"k": k, "hit_rate": len(set(top).intersection(actual[:k])) / k,
                        "mean_experimental_fitness": float(np.mean(observed)),
                        "max_experimental_fitness": float(np.max(observed))})
    return results


def describe_predictions(rows, predictions, cutoff):
    y = np.asarray([row["fitness"] for row in rows], dtype=np.float64)
    p = np.asarray(predictions, dtype=np.float64)
    if p.shape != y.shape or not np.isfinite(p).all():
        raise ValueError("Prediction output must contain one finite number per row")
    counts = np.array([int(row["mutation_count"]) for row in rows])
    masks = ((counts == 1), (counts == 2), (counts == 3),
             (4 <= counts) & (counts <= 5), (6 <= counts) & (counts <= 7),
             (8 <= counts) & (counts <= 15))
    high = y >= cutoff
    return {
        "overall": regression_metrics(y, p, overall=True),
        "high_fitness": regression_metrics(y[high], p[high]),
        "by_mutation_count": {group: regression_metrics(y[mask], p[mask]) for group, mask in zip(GROUPS, masks)},
        "top_k": top_k_metrics(y, p, tie_priority([row[ID] for row in rows])),
    }


def prediction_records(rows, predictions, model, split):
    for row, estimate in zip(rows, predictions):
        residual = float(estimate - float(row["fitness"]))
        yield {ID: row[ID], "mutant": row["mutant"], "mutation_count": row["mutation_count"],
               "experimental_fitness": row["fitness"], "predicted_fitness": float(estimate),
               "residual": residual, "absolute_error": abs(residual), "model_name": model, "split": split}


def error_audit(records, train_fitness, cutoff):
    experimental = np.array([float(row["experimental_fitness"]) for row in records])
    predicted = np.array([float(row["predicted_fitness"]) for row in records])
    residual = predicted - experimental
    train_min, train_max = float(min(train_fitness)), float(max(train_fitness))
    ordered = sorted(records, key=lambda row: row[ID])
    cases = {
        "most_accurate": sorted(ordered, key=lambda row: float(row["absolute_error"]))[:5],
        "overestimated": sorted((r for r in ordered if float(r["residual"]) > 0),
                               key=lambda row: -float(row["residual"]))[:5],
        "underestimated": sorted((r for r in ordered if float(r["residual"]) < 0),
                                key=lambda row: float(row["residual"]))[:5],
    }
    examples = [{"case": kind, "rank": i, **row} for kind, matches in cases.items()
                for i, row in enumerate(matches, start=1)]
    high = experimental >= cutoff
    audit = {
        "experimental_range": [float(experimental.min()), float(experimental.max())],
        "predicted_range": [float(predicted.min()), float(predicted.max())],
        "training_fitness_range": [train_min, train_max],
        "predicted_above_training_max": int((predicted > train_max).sum()),
        "predicted_below_training_min": int((predicted < train_min).sum()),
        "predicted_to_experimental_sd_ratio": float(predicted.std() / experimental.std()) if experimental.std() else None,
        "by_fitness_band": {},
    }
    for name, mask in (("high", high), ("low", ~high)):
        audit["by_fitness_band"][name] = {**regression_metrics(experimental[mask], predicted[mask]),
                                         "mean_residual": float(residual[mask].mean()) if mask.any() else None}
    return examples, audit


def export_metrics(output, metrics):
    main, grouped, top = [], [], []
    for split, families in metrics.items():
        for model, result in families.items():
            key = {"split": split, "model_name": model}
            for scope in ("overall", "high_fitness"):
                main.append({**key, "scope": scope, **result[scope]})
            grouped.extend({**key, "mutation_count_group": group, **numbers}
                           for group, numbers in result["by_mutation_count"].items())
            top.extend({**key, **numbers} for numbers in result["top_k"])
    for name, rows in (("model_metrics", main), ("model_metrics_by_mutation_count", grouped),
                       ("top_k_performance", top)):
        write_csv(output / f"{name}.csv", tuple(rows[0]), rows)


def plot_test_results(output, metrics, records, selected):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 12,
                         "axes.titleweight": "bold", "axes.spines.top": False, "axes.spines.right": False,
                         "axes.axisbelow": True, "grid.color": "#e4ebef", "savefig.dpi": 300,
                         "savefig.facecolor": "white"})

    def save(fig, name):
        fig.savefig(output / f"{name}.png")
        plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6), layout="constrained")
    fig.suptitle("Model performance on the independent test set", weight="bold")
    for ax, metric, name in zip(axes.flat, ("spearman", "pearson", "rmse", "mae"),
                                ("Spearman correlation", "Pearson correlation", "RMSE", "MAE")):
        ax.bar(DISPLAY, [metrics[model]["overall"][metric] for model in MODEL_NAMES], color=COLORS)
        ax.set(ylabel=name)
        ax.grid(axis="y")
    save(fig, "figure_1_model_comparison")

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.9), layout="constrained", sharex=True, sharey=True)
    values = [float(row[field]) for row in records for field in ("experimental_fitness", "predicted_fitness")]
    lower, upper = min(values), max(values)
    for ax, model, name, color in zip(axes, MODEL_NAMES, DISPLAY, COLORS):
        rows = [row for row in records if row["model_name"] == model]
        ax.scatter([float(r["experimental_fitness"]) for r in rows],
                   [float(r["predicted_fitness"]) for r in rows], s=6, alpha=0.20,
                   linewidths=0, color=color)
        ax.plot([lower, upper], [lower, upper], linestyle="--", color="#495560", linewidth=1)
        ax.set(title=f"{name}\nSpearman {metrics[model]['overall']['spearman']:.3f} · "
                     f"RMSE {metrics[model]['overall']['rmse']:.3f}", xlabel="Experimental fitness")
    axes[0].set_ylabel("Predicted fitness")
    save(fig, "figure_2_experimental_vs_predicted")

    fig, ax = plt.subplots(figsize=(7.5, 4.2), layout="constrained")
    ax.hist([float(row["residual"]) for row in records if row["model_name"] == selected],
            bins=40, color=COLORS[MODEL_NAMES.index(selected)], edgecolor="white", linewidth=0.4)
    ax.axvline(0, linestyle="--", color="#495560")
    ax.set(title=f"Prediction residuals — {DISPLAY[MODEL_NAMES.index(selected)]}",
           xlabel="Predicted minus experimental fitness", ylabel="Variant count")
    ax.grid(axis="y")
    save(fig, "figure_3_selected_residuals")

    fig, ax = plt.subplots(figsize=(7.5, 4.2), layout="constrained")
    x = np.arange(len(KS))
    for i, (model, label, color) in enumerate(zip(MODEL_NAMES, DISPLAY, COLORS)):
        ax.bar(x + (i - 1) * 0.24, [row["hit_rate"] for row in metrics[model]["top_k"]],
               width=0.23, color=color, label=label)
    ax.set(title="Top-k hit rate on the independent test set", xlabel="k", ylabel="Top-k hit rate")
    ax.set_xticks(x, [str(k) for k in KS])
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.legend(frameon=False)
    ax.grid(axis="y")
    save(fig, "figure_4_top_k_hit_rate")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), layout="constrained")
    fig.suptitle("Independent test performance by mutation count", weight="bold")
    for ax, metric, title in zip(axes, ("spearman", "rmse"), ("Spearman correlation", "RMSE")):
        for model, label, color in zip(MODEL_NAMES, DISPLAY, COLORS):
            scores = [metrics[model]["by_mutation_count"][group][metric] for group in GROUPS]
            ax.plot(GROUPS, np.array(scores, dtype=float), "o-", color=color, label=label)
        ax.set(xlabel="Substitutions per variant", ylabel=title)
        ax.grid(axis="y")
    axes[0].legend(frameon=False)
    save(fig, "figure_5_performance_by_mutation_count")


def run(root, candidate_smoke=False):
    output = root / "results" / "fitness_model"
    lock = output / "model_selection.json"
    selection = json.loads(lock.read_text(encoding="utf-8"))
    if selection.get("status") != "locked" or selection.get("seed") != SEED:
        raise ValueError("Run train_fitness.py to lock the validation-only selection first")
    for name, digest in selection["locked_files_sha256"].items():
        if file_sha256(root / name) != digest:
            raise ValueError(f"Locked file changed: {name}")
    encoder, cutoff = load_reference(root)
    families = selection["families"]
    models = {name: FitnessPredictor.load(root / families[name]["model_path"]) for name in MODEL_NAMES}
    selected = selection["selected_model"]
    final_model = FitnessPredictor.load(root / selection["selected_model_path"])
    for name, model in models.items():
        if (model.model_name != name or model.params != families[name]["params"]
                or model.encoder.reference != encoder.reference):
            raise ValueError(f"Locked model {name} does not match its configuration")
    training = load_split(root, "train_initial", encoder)
    validation = load_split(root, "validation", encoder)
    # Test labels are opened only after the persisted validation-only choice is checked.
    test = load_split(root, "test", encoder)
    splits = {"train_initial": training, "validation": validation, "test": test}
    if candidate_smoke:
        splits["candidate_pool"] = load_split(root, "candidate_pool", encoder)
    check_disjoint(splits)
    records, test_metrics, timing, fallback = [], {}, {}, {}
    sequences = [row["sequence"] for row in test]
    for name, model in models.items():
        start = perf_counter()
        predictions = model.predict(sequences)
        timing[name] = perf_counter() - start
        if name == "count_only":
            _, levels = model.predict_with_fallback(model.encode(sequences))
            fallback = dict(Counter(levels))
        records.extend(prediction_records(test, predictions, name, "test"))
        test_metrics[name] = describe_predictions(test, predictions, cutoff)

    selected_sample = sequences[:32]
    if not np.allclose(final_model.predict(selected_sample), models[selected].predict(selected_sample),
                       rtol=0, atol=1e-12):
        raise ValueError("Selected model artifact does not match the locked family model")
    candidate_report = None
    if candidate_smoke:
        start = perf_counter()
        estimates = final_model.predict((r["sequence"] for r in splits["candidate_pool"]))
        candidate_report = {"prediction_seconds": perf_counter() - start, "sample_count": len(estimates),
                            "all_predictions_finite": bool(np.isfinite(estimates).all()), "batch_size": 1024}
        if not candidate_report["all_predictions_finite"] or len(estimates) != len(splits["candidate_pool"]):
            raise ValueError("Invalid candidate batch prediction output")
    validation_records = [row for _, row in read_csv(output / "validation_predictions.csv")[1]]
    audit, examples = {}, []
    training_fitness = [float(row["fitness"]) for row in training]
    for split, rows in (("validation", validation_records), ("test", records)):
        cases, audit[split] = error_audit([row for row in rows if row["model_name"] == selected],
                                        training_fitness, cutoff)
        examples.extend(cases)
    write_csv(output / "test_predictions.csv", PRED_FIELDS, records)
    write_csv(output / "prediction_error_analysis.csv", ("case", "rank", *PRED_FIELDS), examples)
    export_metrics(output, {"validation": {name: families[name]["validation"] for name in MODEL_NAMES},
                            "test": test_metrics})
    plot_test_results(output, test_metrics, records, selected)
    hashes = {"data/splits/test.csv": file_sha256(root / "data/splits/test.csv")}
    if candidate_smoke:
        hashes["data/splits/candidate_pool.csv"] = file_sha256(root / "data/splits/candidate_pool.csv")
    write_json(output / "evaluation_summary.json", {
        "sample_counts": {name: len(rows) for name, rows in splits.items()},
        "selected_model": selected, "selection_sha256": file_sha256(lock), "evaluation_input_sha256": hashes,
        "threshold": cutoff, "test_prediction_seconds_including_encoding": timing,
        "count_baseline_fallback": fallback, "test_metrics": test_metrics,
        "selected_error_analysis": audit, "candidate_inference": candidate_report,
        "selected_reload_consistency_test": "passed",
    })
    print(f"Evaluated {len(test):,} independent test variants for {len(models)} locked model families.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--candidate-smoke", action="store_true")
    args = parser.parse_args()
    run(args.root.resolve(), args.candidate_smoke)
