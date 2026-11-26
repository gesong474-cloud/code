"""Build and audit the four GFP splits for virtual directed evolution."""

import argparse
import hashlib
import json
import math
import random
from collections import Counter
from decimal import Decimal
from itertools import combinations
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from matplotlib.ticker import PercentFormatter
from sklearn.model_selection import train_test_split

from data_processing import ASSAY_ID, STANDARD_AA, VARIANT_FIELDS, file_sha256, read_csv, write_csv, write_json
from eda import DARK, summarize

TOTAL = 51_714
DEFAULT_SEED = 42
DEFAULT_SIZES = {"train_initial": 3000, "validation": 1000, "test": 5000, "candidate_pool": 42_714}
SPLITS = tuple(DEFAULT_SIZES)
GROUPS = ("1", "2", "3", "4", "5", "6-7", "8-15")
COUNTS = range(1, 16)
ID = "experiment_variant_id"
CANDIDATE_FIELDS = (
    ID, "mutant", "sequence", "mutation_count", "mutation_positions", "wt_amino_acids", "mutant_amino_acids",
)
LABELED_FIELDS = (*CANDIDATE_FIELDS, "fitness", "fitness_bin")
ORACLE_FIELDS = (ID, "fitness")
MANIFEST_FIELDS = ("original_variant_id", ID, "mutation_count", "stratification_group", "split")
INPUTS = (
    "data/processed/gfp_processed.csv", "data/processed/gfp_wildtype.fasta",
    "data/processed/gfp_metadata.json", "results/eda/data_quality_report.json",
)
DISPLAY_NAMES = ("Initial training", "Validation", "Independent test", "Candidate pool")
COLORS = ("#266794", "#a44931", "#4b7954", "#78588b")


def stratification_group(count):
    if count not in COUNTS:
        raise ValueError(f"Mutation count {count} is outside the assay range 1-15")
    if count <= 5:
        return str(count)
    return "6-7" if count <= 7 else "8-15"


def experiment_variant_id(sequence):
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


def index_variants(rows, wt):
    variants = {}
    original_ids = set()
    for line, row in rows:
        if None in row or any(row[field] is None for field in VARIANT_FIELDS):
            raise ValueError(f"Processed CSV line {line} does not match its schema")
        original_id, sequence = row["variant_id"], row["sequence"]
        if not original_id or original_id in original_ids:
            raise ValueError(f"Missing or duplicate original variant ID at line {line}: {original_id!r}")
        original_ids.add(original_id)
        if len(sequence) != len(wt) or set(sequence) - STANDARD_AA:
            raise ValueError(f"Invalid protein sequence at line {line} ({original_id})")
        identifier = experiment_variant_id(sequence)
        if identifier in variants:
            reason = "duplicate sequence" if variants[identifier]["sequence"] == sequence else "SHA-256 collision"
            raise ValueError(f"Duplicate experiment ID at line {line} ({original_id}): {reason}")
        try:
            count = int(row["mutation_count"])
            group = stratification_group(count)
            finite_fitness = math.isfinite(float(row["fitness"]))
        except ValueError as error:
            raise ValueError(f"Invalid processed data at line {line} ({original_id}): {error}") from error
        if (count != sum(a != b for a, b in zip(wt, sequence))
                or count != len(row["mutant"].split(":"))):
            raise ValueError(f"Mutation count disagrees with sequence or mutant at line {line} ({original_id})")
        if not finite_fitness or row["fitness_bin"] not in {"0", "1"}:
            raise ValueError(f"Invalid fitness or binary label at line {line} ({original_id})")
        variants[identifier] = {**row, ID: identifier, "stratification_group": group}
    return variants


def load_inputs(root):
    assay, fasta, metadata_path, quality_path = (root / name for name in INPUTS)
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    if quality.get("dataset") != ASSAY_ID or quality.get("status") != "passed":
        raise ValueError("Stage 1 quality report must be for the GFP assay and have status 'passed'")
    for field in ("raw_variant_count", "processed_variant_count"):
        if quality.get(field) != TOTAL:
            raise ValueError(f"Stage 1 quality report {field} must equal {TOTAL:,}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    fasta_lines = fasta.read_text(encoding="utf-8").splitlines()
    wt = "".join(fasta_lines[1:])
    if (metadata.get("dataset_name") != ASSAY_ID or metadata.get("reference_total_variants") != TOTAL
            or not fasta_lines or not fasta_lines[0].startswith(f">{ASSAY_ID}|")
            or len(wt) != metadata.get("sequence_length") or len(wt) != 238 or set(wt) - STANDARD_AA):
        raise ValueError("Stage 1 FASTA or metadata does not match the expected GFP assay")
    cutoff = Decimal(metadata["binarization_cutoff_source_value"])
    if cutoff != Decimal("2.5"):
        raise ValueError(f"Expected the ProteinGym GFP cutoff 2.5, found {cutoff}")
    fields, rows = read_csv(assay)
    if tuple(fields) != VARIANT_FIELDS or len(rows) != TOTAL:
        raise ValueError(f"Processed GFP CSV must have the Stage 1 schema and {TOTAL:,} records")
    variants = index_variants(rows, wt)
    if {int(row["mutation_count"]) for row in variants.values()} != set(COUNTS):
        raise ValueError("Processed GFP CSV must contain the complete 1-15 mutation count range")
    return variants, cutoff


def random_streams(seed):
    if seed < 0:
        raise ValueError("Random seed must be a non-negative integer")
    return [int(value) for value in np.random.SeedSequence(seed).generate_state(9)]


def make_splits(groups_by_id, sizes, seed):
    if set(sizes) != set(SPLITS) or sum(sizes.values()) != len(groups_by_id):
        raise ValueError("Split sizes must cover all four subsets and sum to the complete dataset")
    if min(sizes.values()) < len(GROUPS):
        raise ValueError(f"Each split size must be at least {len(GROUPS)} for seven-group stratification")
    if set(groups_by_id.values()) != set(GROUPS):
        raise ValueError("The full dataset must contain all seven stratification groups")
    streams = random_streams(seed)
    # Sorting opaque IDs makes assignment independent of the original input row order.
    remaining = sorted(groups_by_id)
    splits = {}
    for i, name in enumerate(SPLITS[:-1]):
        try:
            splits[name], remaining = train_test_split(
                remaining, train_size=sizes[name], stratify=[groups_by_id[key] for key in remaining],
                random_state=streams[i],
            )
        except ValueError as error:
            raise ValueError(f"Unable to stratify {name} with size {sizes[name]}: {error}") from error
    splits["candidate_pool"] = remaining
    for i, name in enumerate(SPLITS):
        random.Random(streams[3 + i]).shuffle(splits[name])
    return splits


def write_split_tables(directory, variants, splits, seed):
    directory.mkdir(parents=True, exist_ok=True)
    for name, identifiers in splits.items():
        fields = CANDIDATE_FIELDS if name == "candidate_pool" else LABELED_FIELDS
        write_csv(directory / f"{name}.csv", fields,
                  ({field: variants[key][field] for field in fields} for key in identifiers))
    oracle_ids = sorted(splits["candidate_pool"])
    manifest_ids = sorted(variants)
    streams = random_streams(seed)
    random.Random(streams[7]).shuffle(oracle_ids)
    random.Random(streams[8]).shuffle(manifest_ids)
    write_csv(directory / "oracle_labels.csv", ORACLE_FIELDS,
              ({ID: key, "fitness": variants[key]["fitness"]} for key in oracle_ids))
    membership = {key: name for name, identifiers in splits.items() for key in identifiers}
    write_csv(directory / "split_manifest.csv", MANIFEST_FIELDS, (
        {"original_variant_id": variants[key]["variant_id"], ID: key,
         "mutation_count": variants[key]["mutation_count"],
         "stratification_group": variants[key]["stratification_group"], "split": membership[key]}
        for key in manifest_ids
    ))


def load_export(path, fields):
    actual, rows = read_csv(path)
    if tuple(actual) != fields:
        raise ValueError(f"{path.name} has unexpected columns {actual!r}; expected {fields!r}")
    if any(None in row or any(value is None for value in row.values()) for _, row in rows):
        raise ValueError(f"{path.name} contains a row with an invalid number of fields")
    return [row for _, row in rows]


def validate_exports(directory, variants, splits, sizes):
    seen_ids, seen_sequences, actual_membership = set(), set(), {}
    for name in SPLITS:
        fields = CANDIDATE_FIELDS if name == "candidate_pool" else LABELED_FIELDS
        rows = load_export(directory / f"{name}.csv", fields)
        if len(rows) != sizes[name]:
            raise ValueError(f"{name} has {len(rows)} rows; expected {sizes[name]}")
        if [row[ID] for row in rows] != splits[name]:
            raise ValueError(f"{name} does not match the reproducible assignment and order")
        for row in rows:
            key, sequence = row[ID], row["sequence"]
            if key in seen_ids or sequence in seen_sequences:
                raise ValueError(f"Duplicate experiment ID or protein sequence in {name}: {key}")
            if key not in variants or experiment_variant_id(sequence) != key:
                raise ValueError(f"Unexpected experiment ID or sequence in {name}: {key}")
            source = variants[key]
            differences = [field for field in fields if row[field] != source[field]]
            if differences:
                raise ValueError(f"{name} changed source fields {differences} for {key}")
            seen_ids.add(key)
            seen_sequences.add(sequence)
            actual_membership[key] = name
    if seen_ids != set(variants) or seen_sequences != {row["sequence"] for row in variants.values()}:
        raise ValueError("The split union does not equal the complete source dataset")

    oracle = load_export(directory / "oracle_labels.csv", ORACLE_FIELDS)
    candidate_ids = set(splits["candidate_pool"])
    if len(oracle) != len(candidate_ids) or {row[ID] for row in oracle} != candidate_ids:
        raise ValueError("Oracle labels must correspond one-to-one with candidate IDs")
    for row in oracle:
        if row["fitness"] != variants[row[ID]]["fitness"]:
            raise ValueError(f"Oracle fitness differs from source for {row[ID]}")

    manifest = load_export(directory / "split_manifest.csv", MANIFEST_FIELDS)
    if len(manifest) != len(variants) or {row[ID] for row in manifest} != seen_ids:
        raise ValueError("Split manifest must contain each variant exactly once")
    for row in manifest:
        key = row[ID]
        source = variants[key]
        expected = {"original_variant_id": source["variant_id"], ID: key,
                    "mutation_count": source["mutation_count"],
                    "stratification_group": source["stratification_group"], "split": actual_membership[key]}
        if row != expected:
            raise ValueError(f"Incorrect mapping in split manifest for {key}")

    return dict.fromkeys((
        "target_sample_counts", "disjoint_experiment_ids", "disjoint_protein_sequences",
        "exact_input_union", "no_duplicates_or_losses", "candidate_schema_has_no_labels",
        "candidate_oracle_bijection", "source_sequences_mutants_and_fitness_preserved",
        "all_mutation_counts_preserved", "experiment_ids_unique_and_sequence_derived",
        "candidate_and_training_schema_has_no_source_row_id",
        "manifest_mapping_verified",
    ), True)


def validate_reproducibility(groups_by_id, splits, sizes, seed):
    # Reverse the input mapping as well to verify the canonical ID sort.
    reordered = dict(reversed(list(groups_by_id.items())))
    same = make_splits(reordered, sizes, seed)
    if same != splits:
        raise ValueError("Same seed failed to reproduce assignments and row order")
    alternate = make_splits(groups_by_id, sizes, seed + 1)
    all_ids = set(groups_by_id)
    for name in SPLITS:
        ids = alternate[name]
        if len(ids) != sizes[name] or len(set(ids)) != sizes[name]:
            raise ValueError(f"Alternate seed failed target size or uniqueness in {name}")
    if set().union(*(set(keys) for keys in alternate.values())) != all_ids:
        raise ValueError("Alternate seed failed exact partition coverage")
    changed = {name: len(set(splits[name]) - set(alternate[name])) for name in SPLITS}
    if not any(changed.values()):
        raise ValueError("Alternate seed produced identical membership")
    return {"same_seed_identical_assignment_and_order": True,
            "different_seed_distinct_valid_partition": True, "alternate_seed": seed + 1,
            "membership_changes_by_split": changed}


def calculate_statistics(variants, splits, cutoff):
    full = list(variants.values())
    full_scores = np.array([float(row["fitness"]) for row in full])
    full_stats = summarize(full_scores)
    bin_edges = np.linspace(full_scores.min(), full_scores.max(), 41)
    overall_threshold = sum(Decimal(row["fitness"]) >= cutoff for row in full) / len(full)
    full_hist = np.histogram(full_scores, bins=bin_edges)[0] / len(full)
    dimensions = (
        ("exact_mutation_counts", "mutation_count", tuple(map(str, COUNTS))),
        ("stratification_groups", "stratification_group", GROUPS),
    )
    full_counts = {field: Counter(row[field] for row in full) for _, field, _ in dimensions}
    split_stats, scores, distribution = {}, {}, []
    for name, identifiers in splits.items():
        rows = [variants[key] for key in identifiers]
        n = len(rows)
        scores[name] = np.array([float(row["fitness"]) for row in rows])
        n_at_cutoff = sum(Decimal(row["fitness"]) >= cutoff for row in rows)
        summary = {"sample_count": n, "fraction_total": n / len(full),
                   "fitness": summarize(scores[name]), "at_or_above_threshold_count": n_at_cutoff,
                   "at_or_above_threshold_fraction": n_at_cutoff / n}
        for variable, field, categories in dimensions:
            frequencies = Counter(row[field] for row in rows)
            full_freq = full_counts[field]
            summary[variable] = {}
            for category in categories:
                count = frequencies[category]
                record = {"category": category, "count": count, "fraction_of_split": count / n,
                          "fraction_of_full": full_freq[category] / len(full),
                          "delta_percentage_points": 100 * (count / n - full_freq[category] / len(full))}
                distribution.append({"split": name, "distribution": variable, **record})
                summary[variable][category] = {"count": count, "fraction": count / n}
        hist = np.histogram(scores[name], bins=bin_edges)[0] / n
        summary["difference_from_full"] = {
            "mean_fitness_delta": summary["fitness"]["mean"] - full_stats["mean"],
            "median_fitness_delta": summary["fitness"]["median"] - full_stats["median"],
            "threshold_rate_delta_percentage_points": 100 * (n_at_cutoff / n - overall_threshold),
            "fitness_histogram_total_variation": float(0.5 * np.abs(hist - full_hist).sum()),
        }
        split_stats[name] = summary
    return split_stats, distribution, scores, bin_edges


def plot_distributions(output, stats, scores, bin_edges):
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 13,
        "axes.titleweight": "bold", "axes.titlepad": 11, "axes.labelcolor": DARK,
        "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
        "text.color": DARK, "xtick.color": DARK, "ytick.color": DARK,
        "grid.color": "#e4ebef", "grid.linewidth": 0.65, "savefig.dpi": 300,
        "savefig.facecolor": "white",
    })
    fig, axes = plt.subplots(2, 1, figsize=(9.4, 6.7), layout="constrained")
    fig.suptitle("Mutation count distribution across dataset splits", weight="bold", fontsize=15)
    for ax, field, title in zip(axes, ("exact_mutation_counts", "stratification_groups"),
                                ("Exact mutation counts", "Stratification groups")):
        categories = list(stats["train_initial"][field])
        x = np.arange(len(categories))
        for i, (name, label, color) in enumerate(zip(SPLITS, DISPLAY_NAMES, COLORS)):
            values = [stats[name][field][category]["fraction"] for category in categories]
            ax.bar(x + (i - 1.5) * 0.19, values, width=0.18, color=color, label=label)
        ax.set(title=title, xlabel="Substitutions per variant", ylabel="Fraction of split")
        ax.set_xticks(x, categories)
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.grid(axis="y")
    axes[0].legend(ncols=4, frameon=False, loc="upper right")
    fig.savefig(output / "figure_1_mutation_count_distribution.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.4, 4.5), layout="constrained")
    styles = ("-", "--", "-.", ":")
    for name, label, color, style in zip(SPLITS, DISPLAY_NAMES, COLORS, styles):
        values = scores[name]
        ax.hist(values, bins=bin_edges, weights=np.full(len(values), 1 / len(values)),
                histtype="step", linewidth=1.5, color=color, linestyle=style, label=label)
    ax.set(title="Fitness distribution across dataset splits", xlabel="Experimental DMS score",
           ylabel="Fraction of split per bin")
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.grid(axis="y")
    ax.legend(frameon=False, loc="upper right")
    fig.savefig(output / "figure_2_fitness_distribution.png")
    plt.close(fig)


def run(root, seed, sizes):
    variants, cutoff = load_inputs(root)
    groups = {key: row["stratification_group"] for key, row in variants.items()}
    splits = make_splits(groups, sizes, seed)
    reproducibility = validate_reproducibility(groups, splits, sizes, seed)
    with TemporaryDirectory(prefix=".gfp-split-", dir=root) as temporary:
        staging = Path(temporary)
        tables, results = staging / "data/splits", staging / "results/splits"
        write_split_tables(tables, variants, splits, seed)
        checks = validate_exports(tables, variants, splits, sizes)
        stats, distribution, scores, bin_edges = calculate_statistics(variants, splits, cutoff)
        results.mkdir(parents=True)
        write_csv(results / "split_distribution.csv", tuple(distribution[0]), distribution)
        plot_distributions(results, stats, scores, bin_edges)
        max_group_delta = max(abs(row["delta_percentage_points"]) for row in distribution
                              if row["distribution"] == "stratification_groups")
        differences = [entry["difference_from_full"] for entry in stats.values()]
        max_threshold_delta = max(abs(row["threshold_rate_delta_percentage_points"]) for row in differences)
        max_mean_delta = max(abs(row["mean_fitness_delta"]) for row in differences)
        max_tv = max(row["fitness_histogram_total_variation"] for row in differences)
        published = sorted([*tables.iterdir(), *results.iterdir()])
        report = {
            "dataset": ASSAY_ID, "status": "passed", "total_variants": len(variants),
            "random_seed": seed, "target_sizes": sizes, "stratification_labels": list(GROUPS),
            "stratification_variable": "mutation count group",
            "assignment_order": ["train_initial", "validation", "test", "candidate_pool (remainder)"],
            "canonical_input_order": "Lexicographic SHA-256 sequence ID",
            "random_state_derivation": "NumPy SeedSequence; separate streams for allocation and each export order",
            "derived_random_states": random_streams(seed),
            "versions": {"numpy": np.__version__, "scikit_learn": sklearn.__version__,
                         "matplotlib": matplotlib.__version__},
            "reference_sequence_length": 238, "fitness_threshold": {"value": float(cutoff), "comparison": ">="},
            "input_files_sha256": {name: file_sha256(root / name) for name in INPUTS},
            "output_files_sha256": {path.relative_to(staging).as_posix(): file_sha256(path) for path in published},
            "output_hash_scope": "All data split tables, the distribution table and both figures; this summary is excluded.",
            "checks": checks, "reproducibility": reproducibility, "splits": stats,
            "offline_distribution_review": {
                "max_stratification_group_share_delta_percentage_points": max_group_delta,
                "max_threshold_rate_delta_percentage_points": max_threshold_delta,
                "max_abs_mean_fitness_delta": max_mean_delta,
                "max_fitness_histogram_total_variation": max_tv,
                "fitness_histogram_bin_edges": bin_edges.tolist(),
                "label_access": "Offline audit only; do not expose to Agent or prediction models.",
            },
            "future_experiment": {
                "initial_labeled_samples": DEFAULT_SIZES["train_initial"], "evolution_rounds": 3,
                "budget_per_round": 20, "total_additional_budget": 60, "primary_seed": DEFAULT_SEED,
                "oracle": "Experimentally measured DMS scores",
            },
        }
        summary_path = results / "split_summary.json"
        write_json(summary_path, report)
        for path in [*published, summary_path]:
            target = root / path.relative_to(staging)
            target.parent.mkdir(parents=True, exist_ok=True)
            path.replace(target)
    print("Validated partition: " + ", ".join(f"{name}={len(ids):,}" for name, ids in splits.items()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--train-size", type=int, default=DEFAULT_SIZES["train_initial"])
    parser.add_argument("--validation-size", type=int, default=DEFAULT_SIZES["validation"])
    parser.add_argument("--test-size", type=int, default=DEFAULT_SIZES["test"])
    parser.add_argument("--candidate-size", type=int, help="Default: all remaining variants.")
    args = parser.parse_args()
    candidate = args.candidate_size
    if candidate is None:
        candidate = TOTAL - args.train_size - args.validation_size - args.test_size
    sizes = dict(zip(SPLITS, (args.train_size, args.validation_size, args.test_size, candidate)))
    try:
        run(args.root.resolve(), args.seed, sizes)
    except (OSError, ValueError, KeyError, ArithmeticError) as error:
        parser.exit(1, f"Dataset split failed: {error}\n")


if __name__ == "__main__":
    main()
